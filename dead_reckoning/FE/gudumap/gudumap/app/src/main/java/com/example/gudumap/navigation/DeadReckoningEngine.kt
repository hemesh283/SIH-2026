package com.example.gudumap.navigation

import android.content.Context
import android.location.Location
import com.example.gudumap.ml.InputNormalizer
import com.example.gudumap.ml.ModelMetadata
import com.example.gudumap.ml.ModelRunner
import com.example.gudumap.sensor.DiagnosticRecorder
import com.example.gudumap.sensor.ImuSample
import com.example.gudumap.sensor.OrientationSample
import com.example.gudumap.tracking.TrajectoryIntegrator
import com.example.gudumap.tracking.TrajectoryPoint
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * High-level navigation state produced by DeadReckoningEngine.
 */
data class NavigationEngineState(
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val altitude: Double = 0.0,
    val speed: Float = 0f,
    val heading: Float = 0f,
    val distanceTravelled: Double = 0.0,
    val isInitialized: Boolean = false,
    val isStationary: Boolean = false,
    val isBlackout: Boolean = false,
    val timestampNs: Long = 0L
)

/**
 * Production Dead Reckoning Navigation Engine.
 *
 * Architecture Separation:
 * SensorManager -> Preprocessing -> 10 Hz Resampling -> Phone-to-Vehicle Transform
 * -> IMU Buffer (20x6, stride 10) -> InputNormalizer -> ONNX Runtime ModelRunner
 * -> GRU Local Displacement -> Simplified 6-State MVP EKF -> NHC & ZUPT -> MapMatcher -> TrajectoryIntegrator
 *
 * Scope Note:
 * Uses a simplified 6-state MVP navigation filter (position & velocity in NED).
 * Accelerometer/gyroscope biases and attitude errors are not modeled in this MVP state.
 */
class DeadReckoningEngine(
    val modelRunner: ModelRunner = ModelRunner(),
    val normalizer: InputNormalizer = InputNormalizer(),
    val imuBuffer: IMUBuffer = IMUBuffer(),
    val transformer: CoordinateTransformer = CoordinateTransformer(),
    val zuptDetector: ZuptDetector = ZuptDetector(),
    val ekf: EKF = EKF(),
    val nhc: NHC = NHC(),
    val mapMatcher: MapMatcher = PassThroughMapMatcher(),
    val trajectoryIntegrator: TrajectoryIntegrator = TrajectoryIntegrator(),
    val diagnosticRecorder: DiagnosticRecorder = DiagnosticRecorder()
) : AutoCloseable {

    private val stateLock = Any()

    // Geodetic origin (anchor position)
    private var originLat: Double = 0.0
    private var originLon: Double = 0.0
    private var currentLat: Double = 0.0
    private var currentLon: Double = 0.0

    private var currentHeadingDeg: Float = 0f
    private var currentPitchDeg: Float = 0f
    private var currentRollDeg: Float = 0f
    private var latestOrientationMatrix: FloatArray? = null

    private var isBlackoutMode: Boolean = false
    private var isEngineInitialized: Boolean = false
    private var lastStateTimestampNs: Long = 0L
    private var latestGnssSpeed: Float? = null

    // Diagnostic telemetry fields
    private var lastInferenceTimestampNs: Long = 0L
    private var lastInferenceLatencyMs: Float = 0f
    private var lastDisplacementMeters: FloatArray = floatArrayOf(0f, 0f, 0f)

    // Listener for state updates
    var onNavigationStateChanged: ((NavigationEngineState) -> Unit)? = null

    init {
        // Wire IMU buffer window-ready callback to inference and navigation pipeline
        imuBuffer.onWindowReadyListener = { window, timestampNs ->
            processWindowInference(window, timestampNs)
        }
    }

    /**
     * Optional initialization from Android Context (loads ONNX model).
     */
    fun initializeAssets(context: Context) {
        try {
            if (!modelRunner.ready) {
                modelRunner.initializeFromAssets(context)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Start DR from a known GNSS location fix.
     */
    /**
     * Start DR from known coordinates.
     */
    fun initialize(
        latitude: Double,
        longitude: Double,
        speedMps: Float? = null,
        bearingDeg: Float? = null,
        altitude: Double = 0.0
    ) {
        synchronized(stateLock) {
            originLat = latitude
            originLon = longitude
            currentLat = latitude
            currentLon = longitude

            val vNorth = if (speedMps != null && bearingDeg != null) {
                val speed = speedMps.toDouble()
                val bearingRad = Math.toRadians(bearingDeg.toDouble())
                speed * cos(bearingRad)
            } else 0.0

            val vEast = if (speedMps != null && bearingDeg != null) {
                val speed = speedMps.toDouble()
                val bearingRad = Math.toRadians(bearingDeg.toDouble())
                speed * sin(bearingRad)
            } else 0.0

            if (bearingDeg != null) {
                currentHeadingDeg = bearingDeg
            }

            latestGnssSpeed = speedMps

            ekf.initialize(
                pNorth = 0.0,
                pEast = 0.0,
                pDown = 0.0,
                vNorth = vNorth,
                vEast = vEast,
                vDown = 0.0
            )

            imuBuffer.reset()
            zuptDetector.reset()
            trajectoryIntegrator.reset()

            lastStateTimestampNs = System.nanoTime()
            isEngineInitialized = true

            trajectoryIntegrator.addPoint(
                TrajectoryPoint(
                    latitude = currentLat,
                    longitude = currentLon,
                    altitude = altitude,
                    speedMps = speedMps ?: 0f,
                    headingDeg = currentHeadingDeg,
                    timestampNs = lastStateTimestampNs,
                    isBlackout = isBlackoutMode,
                    isStationary = false
                )
            )
        }
        dispatchState()
    }

    /**
     * Start DR from a known GNSS location fix.
     */
    fun initialize(location: Location) {
        val speed = if (location.hasSpeed()) location.speed else null
        val bearing = if (location.hasBearing()) location.bearing else null
        initialize(location.latitude, location.longitude, speed, bearing, location.altitude)
    }

    /**
     * Correct DR using GNSS coordinates.
     */
    fun correctWithGnss(
        latitude: Double,
        longitude: Double,
        speedMps: Float? = null,
        bearingDeg: Float? = null
    ) {
        synchronized(stateLock) {
            if (!isEngineInitialized) {
                initialize(latitude, longitude, speedMps, bearingDeg)
                return
            }

            if (isBlackoutMode) {
                // During blackout, GNSS is only reference for drift evaluation
                return
            }

            if (speedMps != null) {
                latestGnssSpeed = speedMps
            }

            // Approximate NED relative to anchor
            val dLat = Math.toRadians(latitude - originLat)
            val dLon = Math.toRadians(longitude - originLon)
            val latRad = Math.toRadians(originLat)
            val pNorth = dLat * CoordinateTransformer.MEAN_EARTH_RADIUS
            val pEast = dLon * CoordinateTransformer.MEAN_EARTH_RADIUS * cos(latRad)

            ekf.updateGnssPosition(pNorth, pEast, 0.0)

            if (speedMps != null && bearingDeg != null) {
                val speed = speedMps.toDouble()
                val bearingRad = Math.toRadians(bearingDeg.toDouble())
                val vN = speed * cos(bearingRad)
                val vE = speed * sin(bearingRad)
                ekf.updateGnssVelocity(vN, vE, 0.0)
                currentHeadingDeg = bearingDeg
            }

            // Smoothly fuse GNSS position and velocity into EKF state.
            // Do NOT abruptly snap or overwrite currentLat/currentLon with raw GNSS coordinates;
            // instead derive the position from the updated Kalman filter state.
            val newGeodetic = transformer.addMetricDisplacementToGeodetic(
                originLat, originLon,
                ekf.state[0], // p_N after Kalman update
                ekf.state[1]  // p_E after Kalman update
            )
            currentLat = newGeodetic[0]
            currentLon = newGeodetic[1]
        }
        dispatchState()
    }

    /**
     * Correct DR using GNSS fix (when GNSS is active outside blackout).
     */
    fun correctWithGnss(location: Location) {
        val speed = if (location.hasSpeed()) location.speed else null
        val bearing = if (location.hasBearing()) location.bearing else null
        correctWithGnss(location.latitude, location.longitude, speed, bearing)
    }

    /**
     * Feed incoming raw IMU sample into the pipeline.
     */
    fun addSensorSample(sample: ImuSample) {
        if (!isEngineInitialized) return

        // 1. Phone-to-vehicle coordinate transformation
        val phoneAcc = floatArrayOf(sample.ax, sample.ay, sample.az)
        val vehAcc = transformer.transformPhoneToVehicle(phoneAcc)

        val phoneGyro = floatArrayOf(sample.gx, sample.gy, sample.gz)
        val vehGyro = transformer.transformPhoneToVehicle(phoneGyro)

        val transformedSample = ImuSample(
            timestampNs = sample.timestampNs,
            ax = vehAcc[0],
            ay = vehAcc[1],
            az = vehAcc[2],
            gx = vehGyro[0],
            gy = vehGyro[1],
            gz = vehGyro[2]
        )

        // 2. ZUPT detection (multi-signal: accel, gyro, optional GNSS speed)
        // Strictly isolate GNSS speed during blackout mode to prevent ground-truth leakage
        val gnssSpeedForZupt = if (isBlackoutMode) null else latestGnssSpeed
        val isStationary = zuptDetector.update(transformedSample, gnssSpeedForZupt)
        if (isStationary && zuptDetector.isEnabled) {
            // Pseudo-measurement update in EKF (never hard-clamp velocity!)
            ekf.updateZupt()
        }

        // 3. Feed into 10 Hz resampler & IMU Buffer
        imuBuffer.addSample(transformedSample)

        // 4. Record diagnostic frame if recorder is enabled
        diagnosticRecorder.recordSample(
            sample = transformedSample,
            bufferSize = imuBuffer.size,
            headingDeg = currentHeadingDeg,
            isGnssAvailable = (latestGnssSpeed != null && !isBlackoutMode),
            isBlackout = isBlackoutMode,
            lastInferenceTimestampNs = lastInferenceTimestampNs,
            lastInferenceLatencyMs = lastInferenceLatencyMs,
            lastDisplacement = lastDisplacementMeters
        )
    }

    /**
     * Feed incoming orientation sample.
     */
    fun updateOrientation(orientation: OrientationSample) {
        synchronized(stateLock) {
            currentHeadingDeg = orientation.headingDegrees
            currentPitchDeg = Math.toDegrees(orientation.pitchRad.toDouble()).toFloat()
            currentRollDeg = Math.toDegrees(orientation.rollRad.toDouble()).toFloat()
            latestOrientationMatrix = orientation.rotationMatrix.clone()
        }
    }

    /**
     * Process ready 20x6 window (scheduled at stride = 10 samples).
     */
    private fun processWindowInference(window: Array<FloatArray>, timestampNs: Long) {
        if (!isEngineInitialized) return

        val t0 = System.nanoTime()
        // 1. Run ONNX Model (or fallback mock if session not loaded)
        val localDisplacement = if (modelRunner.ready) {
            try {
                modelRunner.predict(window)
            } catch (e: Exception) {
                // Fallback displacement in case of inference error
                floatArrayOf(0f, 0f, 0f)
            }
        } else {
            // Deterministic nominal integration if model asset is uninitialized
            floatArrayOf(0f, 0f, 0f)
        }
        val t1 = System.nanoTime()
        val latencyMs = (t1 - t0) / 1_000_000f

        synchronized(stateLock) {
            lastInferenceTimestampNs = timestampNs
            lastInferenceLatencyMs = latencyMs
            lastDisplacementMeters = localDisplacement.clone()

            // 2. Coordinate transformation:
            // R_start * delta_p_local -> delta_p_world (NED)
            val rotMatrix = latestOrientationMatrix
            val deltaNed = if (rotMatrix != null && rotMatrix.size == 9) {
                transformer.rotateLocalToWorldWithMatrix(localDisplacement, rotMatrix)
            } else {
                transformer.rotateLocalToWorld(localDisplacement, currentHeadingDeg)
            }

            val deltaNedDouble = doubleArrayOf(
                deltaNed[0].toDouble(),
                deltaNed[1].toDouble(),
                deltaNed[2].toDouble()
            )

            // 3. EKF prediction step
            val dt = ModelMetadata.STRIDE_DURATION_SEC.toDouble() // 1.0s stride duration
            ekf.predict(deltaNedDouble, dt)

            // 4. Non-Holonomic Constraints (NHC)
            if (nhc.isEnabled) {
                nhc.applyConstraint(ekf, currentHeadingDeg, currentPitchDeg)
            }

            // 5. Update geodetic coordinates from EKF position
            val newGeodetic = transformer.addMetricDisplacementToGeodetic(
                originLat, originLon,
                ekf.state[0], // p_N
                ekf.state[1]  // p_E
            )
            currentLat = newGeodetic[0]
            currentLon = newGeodetic[1]

            // Calculate heading from velocity if moving
            val vN = ekf.state[3]
            val vE = ekf.state[4]
            val speed = sqrt(vN * vN + vE * vE).toFloat()

            if (speed > 0.5f) {
                var calcHeading = Math.toDegrees(atan2(vE, vN)).toFloat()
                if (calcHeading < 0f) calcHeading += 360f
                currentHeadingDeg = calcHeading
            }

            lastStateTimestampNs = timestampNs

            // 6. Trajectory integrator with map-matching abstraction
            val rawPoint = TrajectoryPoint(
                latitude = currentLat,
                longitude = currentLon,
                altitude = -ekf.state[2],
                speedMps = speed,
                headingDeg = currentHeadingDeg,
                timestampNs = timestampNs,
                isBlackout = isBlackoutMode,
                isStationary = zuptDetector.isStationary
            )
            val point = mapMatcher.match(rawPoint)
            trajectoryIntegrator.addPoint(point)
        }

        dispatchState()
    }

    fun setBlackoutMode(enabled: Boolean) {
        synchronized(stateLock) {
            isBlackoutMode = enabled
            if (enabled) {
                latestGnssSpeed = null // strictly isolate GNSS ground truth during blackout
                trajectoryIntegrator.startBlackout(
                    TrajectoryPoint(
                        latitude = currentLat,
                        longitude = currentLon,
                        speedMps = getState().speed,
                        headingDeg = currentHeadingDeg,
                        isBlackout = true
                    )
                )
            }
        }
    }

    fun getState(): NavigationEngineState {
        synchronized(stateLock) {
            val vN = ekf.state[3]
            val vE = ekf.state[4]
            val speed = sqrt(vN * vN + vE * vE).toFloat()

            return NavigationEngineState(
                latitude = currentLat,
                longitude = currentLon,
                altitude = -ekf.state[2],
                speed = speed,
                heading = currentHeadingDeg,
                distanceTravelled = trajectoryIntegrator.getTotalDistance(),
                isInitialized = isEngineInitialized,
                isStationary = zuptDetector.isStationary,
                isBlackout = isBlackoutMode,
                timestampNs = lastStateTimestampNs
            )
        }
    }

    private fun dispatchState() {
        val st = getState()
        onNavigationStateChanged?.invoke(st)
    }

    fun reset() {
        synchronized(stateLock) {
            originLat = 0.0
            originLon = 0.0
            currentLat = 0.0
            currentLon = 0.0
            currentHeadingDeg = 0f
            currentPitchDeg = 0f
            currentRollDeg = 0f
            latestOrientationMatrix = null
            isBlackoutMode = false
            isEngineInitialized = false
            latestGnssSpeed = null
            lastInferenceTimestampNs = 0L
            lastInferenceLatencyMs = 0f
            lastDisplacementMeters = floatArrayOf(0f, 0f, 0f)
            diagnosticRecorder.clear()
            imuBuffer.reset()
            zuptDetector.reset()
            ekf.reset()
            trajectoryIntegrator.reset()
        }
    }

    override fun close() {
        modelRunner.close()
    }
}
