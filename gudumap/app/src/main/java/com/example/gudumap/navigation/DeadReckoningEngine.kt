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
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt

enum class GateAction {
    ACCEPTED,
    CLAMPED,
    REJECTED
}

/**
 * High-level navigation state produced by DeadReckoningEngine.
 */
data class NavigationEngineState(
    val latitude: Double = 0.0, // 0.0/0.0 is a "no real fix yet" sentinel -- see isInitialized
    val longitude: Double = 0.0,
    val altitude: Double = 0.0,
    val speed: Float = 0f,
    val heading: Float = 0f,
    val distanceTravelled: Double = 0.0,
    val isInitialized: Boolean = false,
    val isStationary: Boolean = false,
    val motionState: String = "STATIONARY",
    val isBlackout: Boolean = false,
    val timestampNs: Long = 0L,
    val latestGateAction: String = "ACCEPTED",
    val acceptedCount: Int = 0,
    val clampedCount: Int = 0,
    val rejectedCount: Int = 0,
    val naiveLatitude: Double = 0.0,
    val naiveLongitude: Double = 0.0,
    val uncertaintyRadiusMeters: Double = 0.0
)

/**
 * Production Dead Reckoning Navigation Engine.
 *
 * Architecture Separation:
 * SensorManager -> Preprocessing -> 10 Hz Resampling -> Phone-to-Vehicle Transform
 * -> IMU Buffer (20x6, stride 10) -> InputNormalizer -> ONNX Runtime ModelRunner
 * -> GRU Local Displacement -> Simplified 6-State MVP EKF -> NHC & ZUPT -> MapMatcher -> TrajectoryIntegrator
 */
class DeadReckoningEngine(
    val modelRunner: ModelRunner = ModelRunner(),
    val normalizer: InputNormalizer = InputNormalizer(),
    val imuBuffer: IMUBuffer = IMUBuffer(),
    val transformer: CoordinateTransformer = CoordinateTransformer(),
    val zuptDetector: ZuptDetector = ZuptDetector(),
    val ekf: EKF = EKF(),
    val nhc: NHC = NHC(),
    var mapMatcher: MapMatcher = PassThroughMapMatcher(),
    val trajectoryIntegrator: TrajectoryIntegrator = TrajectoryIntegrator(),
    val diagnosticRecorder: DiagnosticRecorder = DiagnosticRecorder(),
    val naiveIntegrator: NaiveIntegrator = NaiveIntegrator()
) : AutoCloseable {

    companion object {
        // Independent, non-self-referential bound for the ML kinematic gate during blackout
        // (PROJECT_STATUS.md §11 Fix 1). The gate previously used the EKF's own live velocity
        // as its "how fast could we plausibly be going" reference, which an accepted/clamped
        // correction could itself raise -- a feedback loop. These grow the ceiling only from a
        // real prior speed and elapsed time, never from the filter's own output.
        private const val MAX_SPEED_CHANGE_MPS2 = 4.0f   // plausible vehicle accel/decel envelope, m/s^2
        private const val MAX_PLAUSIBLE_SPEED_MPS = 50.0f // absolute sanity ceiling (~180 km/h), m/s
    }

    private val stateLock = Any()

    // Geodetic origin (anchor position). 0.0/0.0 is a deliberate "no real fix yet" sentinel
    // (Fix 2, PROJECT_STATUS.md §11) -- not a real-looking coordinate, so nothing downstream
    // can mistake it for an actual position. isEngineInitialized stays false until a real
    // initialize() call (from a genuine GNSS fix) replaces these.
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

    // Speed sanitization (PROJECT_STATUS.md §15) -- a single spurious raw GPS/Network speed
    // reading (a known indoor artifact) was found to corrupt BOTH the EKF velocity state
    // (displayed Speed, Finding A) and Fix 1's blackoutEntrySpeedMps baseline (Finding B) via
    // the same unfiltered value. Bounds how fast a reported speed may change between fixes,
    // using the same physically-motivated ceiling Fix 1 already uses.
    private var lastSanitizedGnssSpeed: Float = 0f
    private var lastGnssSpeedTimestampNs: Long = 0L

    // Independent reference for the ML kinematic gate during blackout (Fix 1) -- captured once
    // at blackout entry from the last real GNSS speed, decoupled from the EKF's own velocity.
    private var blackoutEntrySpeedMps: Float = 0f
    private var blackoutEntryTimestampNs: Long = 0L

    // Diagnostic telemetry fields
    private var lastInferenceTimestampNs: Long = 0L
    private var lastInferenceLatencyMs: Float = 0f
    private var lastDisplacementMeters: FloatArray = floatArrayOf(0f, 0f, 0f)

    // ML Kinematic Gate tracking
    private var currentGateAction: GateAction = GateAction.ACCEPTED
    private var acceptedPredictionCount: Int = 0
    private var clampedPredictionCount: Int = 0
    private var rejectedPredictionCount: Int = 0

    val acceptedCount: Int get() = synchronized(stateLock) { acceptedPredictionCount }
    val clampedCount: Int get() = synchronized(stateLock) { clampedPredictionCount }
    val rejectedCount: Int get() = synchronized(stateLock) { rejectedPredictionCount }
    val latestGateAction: GateAction get() = synchronized(stateLock) { currentGateAction }
    val blackoutMode: Boolean get() = synchronized(stateLock) { isBlackoutMode }

    // Smooth recovery re-anchoring
    private var reanchorRemainingN = 0.0
    private var reanchorRemainingE = 0.0
    private var isSmoothReanchoringActive = false

    val isReanchoring: Boolean get() = synchronized(stateLock) { isSmoothReanchoringActive }

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
            naiveIntegrator.reset()

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
     * Correct DR using GNSS fix.
     */
    fun correctWithGnss(location: Location) {
        val rawSpeed = if (location.hasSpeed()) location.speed else null
        val bearing = if (location.hasBearing()) location.bearing else null
        val speed = sanitizeGnssSpeed(rawSpeed, location.elapsedRealtimeNanos)
        correctWithGnss(location.latitude, location.longitude, speed, bearing)
    }

    /**
     * Bounds how fast a reported GNSS speed may change between consecutive fixes, using the
     * same physically-motivated acceleration ceiling as the kinematic gate (Fix 1). A single
     * spurious raw reading (a known Network-provider/indoor artifact -- derived from noisy
     * consecutive position fixes divided by elapsed time) previously reached both
     * `latestGnssSpeed`/`blackoutEntrySpeedMps` and `ekf.updateGnssVelocity()` completely
     * unfiltered; this is the one place both of those now draw from instead
     * (PROJECT_STATUS.md §15).
     */
    private fun sanitizeGnssSpeed(rawSpeedMps: Float?, timestampNs: Long): Float? {
        if (rawSpeedMps == null) return null

        if (lastGnssSpeedTimestampNs == 0L) {
            // First reading ever (or since reset) -- nothing to compare against, accept as-is.
            lastGnssSpeedTimestampNs = timestampNs
            lastSanitizedGnssSpeed = rawSpeedMps
            return rawSpeedMps
        }

        val dt = max(0.001, (timestampNs - lastGnssSpeedTimestampNs) / 1_000_000_000.0).toFloat()
        lastGnssSpeedTimestampNs = timestampNs

        val maxDelta = MAX_SPEED_CHANGE_MPS2 * dt
        val delta = rawSpeedMps - lastSanitizedGnssSpeed
        val sanitized = if (abs(delta) > maxDelta) {
            lastSanitizedGnssSpeed + maxDelta * sign(delta)
        } else {
            rawSpeedMps
        }

        lastSanitizedGnssSpeed = sanitized
        return sanitized
    }

    /**
     * Public entry point for sanitizing a raw GNSS speed reading, for callers outside this
     * class that need the same bound logic `correctWithGnss` applies internally -- reuses
     * [sanitizeGnssSpeed] rather than duplicating its math. Shares that method's state
     * (`lastSanitizedGnssSpeed`/`lastGnssSpeedTimestampNs`), so calling this also advances the
     * baseline the *next* `correctWithGnss` call will compare against -- see PROJECT_STATUS.md
     * §17 for why that's correct here (the caller is seeding blackout-entry velocity from the
     * same real fix `correctWithGnss` already processed moments earlier, not a fix it hasn't
     * seen yet). Currently used by `NavigationEngine.setBlackoutMode(true)`.
     */
    fun sanitizeExternalGnssSpeed(rawSpeedMps: Float?, timestampNs: Long): Float? {
        return sanitizeGnssSpeed(rawSpeedMps, timestampNs)
    }

    /**
     * Feed incoming raw IMU sample into the pipeline.
     */
    fun addSensorSample(sample: ImuSample) {
        if (!isEngineInitialized) {
            isEngineInitialized = true
            ekf.initialize(0.0, 0.0, 0.0)
        }

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
        val gnssSpeedForZupt = if (isBlackoutMode) null else latestGnssSpeed
        val isStationary = zuptDetector.update(transformedSample, gnssSpeedForZupt)
        if (isStationary && zuptDetector.isEnabled) {
            // Apply ZUPT pseudo-measurement in EKF
            ekf.updateZupt()
        }

        // 3. Feed into 10 Hz resampler & IMU Buffer
        imuBuffer.addSample(transformedSample)

        // 3b. Naive (uncorrected) double-integration, purely for visual contrast --
        // no ZUPT/ML/EKF involved, so it is expected to drift badly on its own.
        val worldAcc = transformer.rotateLocalToWorld(vehAcc, currentHeadingDeg)
        naiveIntegrator.addSample(sample.timestampNs, worldAcc[0], worldAcc[1])

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

        // 1. Motion state and statistics from incoming IMU window
        val currentMotionState = zuptDetector.motionState
        val isNavStationary = zuptDetector.isNavStationary

        var maxHorizAcc = 0f
        var sumHorizAcc = 0f
        var maxTotalAcc = 0f
        var maxGyro = 0f
        for (row in window) {
            val ax = row[0] * ModelMetadata.GRAVITY_MPS2
            val ay = row[1] * ModelMetadata.GRAVITY_MPS2
            val az = row[2] * ModelMetadata.GRAVITY_MPS2
            val aH = sqrt(ax * ax + ay * ay)
            val aTot = sqrt(ax * ax + ay * ay + az * az)
            val gMag = sqrt(row[3] * row[3] + row[4] * row[4] + row[5] * row[5])
            if (aH > maxHorizAcc) maxHorizAcc = aH
            if (aTot > maxTotalAcc) maxTotalAcc = aTot
            if (gMag > maxGyro) maxGyro = gMag
            sumHorizAcc += aH
        }

        // Current filter velocities before update (still used for logging/non-blackout gating)
        val vN_before = ekf.state[3]
        val vE_before = ekf.state[4]
        val vBefore = sqrt(vN_before * vN_before + vE_before * vE_before).toFloat()

        // Kinematic gate reference speed. Outside blackout, GNSS keeps correcting the EKF, so
        // using its live velocity is safe. During blackout, using the EKF's own velocity here
        // would be self-referential -- an accepted/clamped correction raises ekf.state[3]/[4],
        // which would then raise the ceiling that gates the *next* correction, with nothing
        // independent bounding the loop (Fix 1, PROJECT_STATUS.md §11). Instead, during
        // blackout, bound the reference to an envelope grown only from the last known real
        // speed at blackout entry and elapsed blackout time -- both fixed once blackout starts,
        // untouched by anything the ML model or EKF does afterward.
        val baseSpeed = if (isBlackoutMode) {
            if (blackoutEntryTimestampNs == 0L) {
                blackoutEntryTimestampNs = timestampNs
            }
            val elapsedBlackoutSec = max(0.0, (timestampNs - blackoutEntryTimestampNs) / 1_000_000_000.0).toFloat()
            val speedEnvelope = blackoutEntrySpeedMps + MAX_SPEED_CHANGE_MPS2 * elapsedBlackoutSec
            min(speedEnvelope, MAX_PLAUSIBLE_SPEED_MPS)
        } else {
            max(vBefore, latestGnssSpeed ?: 0f)
        }
        val dt = ModelMetadata.STRIDE_DURATION_SEC.toDouble() // 1.0s stride duration
        val dtF = dt.toFloat()

        // 2. ML Kinematic Plausibility Gate
        var gateAction = GateAction.ACCEPTED
        var maxPlausibleDist = 0f
        var localDisplacement = floatArrayOf(0f, 0f, 0f)
        var rawMag = 0f

        if (isNavStationary) {
            // Stationary on table/hand, or rotating in place without horizontal translation
            gateAction = GateAction.REJECTED
            maxPlausibleDist = 0.0f
            localDisplacement = floatArrayOf(0f, 0f, 0f)
        } else if (modelRunner.ready) {
            val rawPred = try {
                modelRunner.predict(window)
            } catch (e: Exception) {
                floatArrayOf(0f, 0f, 0f)
            }
            rawMag = sqrt(rawPred[0] * rawPred[0] + rawPred[1] * rawPred[1] + rawPred[2] * rawPred[2])

            // Kinematic displacement bound: s_kinematic = v0 * dt + 0.5 * a_max * dt^2
            val effectiveAcc = max(maxHorizAcc, 0.20f)
            val kinematicDist = baseSpeed * dtF + 0.5f * effectiveAcc * dtF * dtF
            val gateTolerance = 1.2f
            maxPlausibleDist = kinematicDist + gateTolerance

            if (maxHorizAcc < 0.35f && baseSpeed < 0.30f && rawMag > maxPlausibleDist) {
                // No convincing translational acceleration and near-zero starting speed
                // Large prediction is rejected as an implausible ML measurement
                gateAction = GateAction.REJECTED
                localDisplacement = floatArrayOf(0f, 0f, 0f)
            } else if (rawMag > maxPlausibleDist) {
                // Legitimate motion, but ML predicted beyond kinematic bounds: clamp to physical kinematics
                gateAction = GateAction.CLAMPED
                val maxClampedDist = kinematicDist + 0.15f
                val scale = if (rawMag > 0.001f) (maxClampedDist / rawMag) else 0f
                localDisplacement = floatArrayOf(rawPred[0] * scale, rawPred[1] * scale, rawPred[2] * scale)
            } else {
                gateAction = GateAction.ACCEPTED
                localDisplacement = rawPred
            }
        }
        val t1 = System.nanoTime()
        val latencyMs = (t1 - t0) / 1_000_000f

        synchronized(stateLock) {
            currentGateAction = gateAction
            when (gateAction) {
                GateAction.ACCEPTED -> acceptedPredictionCount++
                GateAction.CLAMPED -> clampedPredictionCount++
                GateAction.REJECTED -> rejectedPredictionCount++
            }

            lastInferenceTimestampNs = timestampNs
            lastInferenceLatencyMs = latencyMs
            lastDisplacementMeters = localDisplacement.clone()

            // 3. EKF prediction step
            if (isNavStationary || gateAction == GateAction.REJECTED) {
                // When rejected or stationary, feed [0,0,0] to EKF and enforce zero-velocity
                ekf.predict(doubleArrayOf(0.0, 0.0, 0.0), dt)
                ekf.updateZupt()
                ekf.state[3] = 0.0
                ekf.state[4] = 0.0
                ekf.state[5] = 0.0
            } else {
                // 3b. Coordinate transformation: local displacement -> world NED
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
                ekf.predict(deltaNedDouble, dt)

                // 4. Non-Holonomic Constraints (NHC)
                if (nhc.isEnabled) {
                    nhc.applyConstraint(ekf, currentHeadingDeg, currentPitchDeg)
                }
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
            var speed = sqrt(vN * vN + vE * vE).toFloat()

            if (isNavStationary || speed < 0.1f) {
                speed = 0f
            } else if (speed > 0.5f) {
                var calcHeading = Math.toDegrees(atan2(vE, vN)).toFloat()
                if (calcHeading < 0f) calcHeading += 360f
                currentHeadingDeg = calcHeading
            }

            lastStateTimestampNs = timestampNs

            val prevDist = trajectoryIntegrator.getTotalDistance()
            val rawPoint = TrajectoryPoint(
                latitude = currentLat,
                longitude = currentLon,
                altitude = -ekf.state[2],
                speedMps = speed,
                headingDeg = currentHeadingDeg,
                timestampNs = timestampNs,
                isBlackout = isBlackoutMode,
                isStationary = isNavStationary
            )
            val point = mapMatcher.match(rawPoint)
            trajectoryIntegrator.addPoint(point)
            val newDist = trajectoryIntegrator.getTotalDistance()
            val distIncr = newDist - prevDist

            try {
                android.util.Log.i("GUDUMAP_DIAG", String.format(
                    "ML_GATE: predicted=%.2fm, max=%.2fm, action=%s, state=%s | |a_h|=%.2f |w|=%.2f | EKF_V_before=%.2f EKF_V_after=%.2f | dDist=%.2f dist=%.2f",
                    rawMag, maxPlausibleDist, gateAction.name, currentMotionState.name,
                    maxHorizAcc, maxGyro,
                    vBefore, speed,
                    distIncr, newDist
                ))
            } catch (_: Throwable) {}
        }

        dispatchState()
    }

    /**
     * Legacy Euler / manual update support for high-frequency or unit tests.
     */
    fun update(
        accelNorth: Float,
        accelEast: Float,
        headingDeg: Float,
        timestampNs: Long = System.nanoTime(),
        isStationary: Boolean = false
    ): NavigationEngineState {
        synchronized(stateLock) {
            if (!isEngineInitialized) {
                initialize(originLat, originLon)
            }

            currentHeadingDeg = headingDeg
            if (isStationary) {
                zuptDetector.reset()
                for (i in 0 until 20) {
                    zuptDetector.update(ImuSample(timestampNs, 0f, 0f, 0f, 0f, 0f, 0f), 0f)
                }
                ekf.updateZupt()
                return getState()
            } else {
                val accMag = sqrt(accelNorth * accelNorth + accelEast * accelEast)
                if (accMag > 0.2f) {
                    zuptDetector.update(ImuSample(timestampNs, accelNorth, accelEast, 0f, 0f, 0f, 0f), 1.5f)
                }
            }

            val dt = if (lastStateTimestampNs > 0L && timestampNs > lastStateTimestampNs) {
                max(0.001, (timestampNs - lastStateTimestampNs) / 1_000_000_000.0)
            } else {
                0.01
            }
            lastStateTimestampNs = timestampNs

            // Calculate delta position from acceleration
            val vN = ekf.state[3] + accelNorth * dt
            val vE = ekf.state[4] + accelEast * dt
            val dN = vN * dt
            val dE = vE * dt

            ekf.predict(doubleArrayOf(dN, dE, 0.0), dt)

            val newGeodetic = transformer.addMetricDisplacementToGeodetic(
                originLat, originLon,
                ekf.state[0],
                ekf.state[1]
            )
            currentLat = newGeodetic[0]
            currentLon = newGeodetic[1]

            val speed = sqrt(ekf.state[3] * ekf.state[3] + ekf.state[4] * ekf.state[4]).toFloat()

            trajectoryIntegrator.addPoint(
                TrajectoryPoint(
                    latitude = currentLat,
                    longitude = currentLon,
                    altitude = -ekf.state[2],
                    speedMps = speed,
                    headingDeg = headingDeg,
                    timestampNs = timestampNs,
                    isBlackout = isBlackoutMode,
                    isStationary = false
                )
            )

            return getState()
        }
    }

    /**
     * Legacy ML displacement update support for direct displacement injection.
     */
    fun updateWithMLDisplacement(
        globalEastMeters: Float,
        globalNorthMeters: Float,
        dtSeconds: Float,
        headingDeg: Float,
        isStationary: Boolean = false
    ): NavigationEngineState {
        synchronized(stateLock) {
            if (!isEngineInitialized) {
                initialize(originLat, originLon)
            }

            currentHeadingDeg = headingDeg
            if (isStationary) {
                zuptDetector.reset()
                for (i in 0 until 20) {
                    zuptDetector.update(ImuSample(System.nanoTime(), 0f, 0f, 0f, 0f, 0f, 0f), 0f)
                }
                ekf.updateZupt()
                return getState()
            } else {
                val dispMag = sqrt(globalEastMeters * globalEastMeters + globalNorthMeters * globalNorthMeters)
                if (dispMag > 0.1f) {
                    zuptDetector.update(ImuSample(System.nanoTime(), 0.8f, 0.2f, 0.5f, 0.1f, 0.1f, 0.1f), 1.5f)
                }
            }

            val dt = dtSeconds.toDouble()
            val deltaNed = doubleArrayOf(globalNorthMeters.toDouble(), globalEastMeters.toDouble(), 0.0)
            ekf.predict(deltaNed, dt)

            val newGeodetic = transformer.addMetricDisplacementToGeodetic(
                originLat, originLon,
                ekf.state[0],
                ekf.state[1]
            )
            currentLat = newGeodetic[0]
            currentLon = newGeodetic[1]

            val vN = ekf.state[3]
            val vE = ekf.state[4]
            val speed = sqrt(vN * vN + vE * vE).toFloat()

            trajectoryIntegrator.addPoint(
                TrajectoryPoint(
                    latitude = currentLat,
                    longitude = currentLon,
                    altitude = -ekf.state[2],
                    speedMps = speed,
                    headingDeg = headingDeg,
                    timestampNs = System.nanoTime(),
                    isBlackout = isBlackoutMode,
                    isStationary = false
                )
            )

            return getState()
        }
    }

    fun setBlackoutMode(enabled: Boolean) {
        synchronized(stateLock) {
            isBlackoutMode = enabled
            if (enabled) {
                // Capture the last known real speed BEFORE isolating GNSS ground truth -- this
                // becomes the kinematic gate's independent reference for the whole blackout
                // (Fix 1). blackoutEntryTimestampNs is captured lazily on the first window
                // processed during this blackout, so it shares the same clock as `timestampNs`
                // in processWindowInference rather than mixing in System.nanoTime().
                blackoutEntrySpeedMps = latestGnssSpeed ?: 0f
                blackoutEntryTimestampNs = 0L
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
            } else {
                blackoutEntrySpeedMps = 0f
                blackoutEntryTimestampNs = 0L
            }
        }
    }

    fun recordGateAction(action: GateAction) {
        synchronized(stateLock) {
            currentGateAction = action
            when (action) {
                GateAction.ACCEPTED -> acceptedPredictionCount++
                GateAction.CLAMPED -> clampedPredictionCount++
                GateAction.REJECTED -> rejectedPredictionCount++
            }
        }
    }

    /**
     * Start smooth re-anchoring transition towards a target GNSS coordinate.
     * Computes the metric error offset and initializes damped step convergence.
     */
    fun startSmoothReanchoring(targetLat: Double, targetLon: Double) {
        synchronized(stateLock) {
            val dLat = Math.toRadians(targetLat - currentLat)
            val dLon = Math.toRadians(targetLon - currentLon)
            val latRad = Math.toRadians(currentLat)
            reanchorRemainingN = dLat * CoordinateTransformer.MEAN_EARTH_RADIUS
            reanchorRemainingE = dLon * CoordinateTransformer.MEAN_EARTH_RADIUS * cos(latRad)
            isSmoothReanchoringActive = true
        }
    }

    private fun stepSmoothReanchoringLocked(): Boolean {
        if (!isSmoothReanchoringActive) return false

        // Rate-limited smooth adjustment: max 1.0 m per step (or 25% of remaining)
        val stepFraction = 0.25
        var corrN = reanchorRemainingN * stepFraction
        var corrE = reanchorRemainingE * stepFraction
        val corrMag = sqrt(corrN * corrN + corrE * corrE)
        val maxStepDist = 1.0 // maximum 1.0m shift per step to strictly prevent jumps
        if (corrMag > maxStepDist) {
            val scale = maxStepDist / corrMag
            corrN *= scale
            corrE *= scale
        }

        ekf.state[0] += corrN
        ekf.state[1] += corrE

        reanchorRemainingN -= corrN
        reanchorRemainingE -= corrE

        val newGeodetic = transformer.addMetricDisplacementToGeodetic(
            originLat, originLon,
            ekf.state[0],
            ekf.state[1]
        )
        currentLat = newGeodetic[0]
        currentLon = newGeodetic[1]

        val remainingDist = sqrt(reanchorRemainingN * reanchorRemainingN + reanchorRemainingE * reanchorRemainingE)
        if (remainingDist < 0.08) {
            // Apply final residual and complete re-anchoring
            ekf.state[0] += reanchorRemainingN
            ekf.state[1] += reanchorRemainingE
            reanchorRemainingN = 0.0
            reanchorRemainingE = 0.0
            val finalGeodetic = transformer.addMetricDisplacementToGeodetic(
                originLat, originLon,
                ekf.state[0],
                ekf.state[1]
            )
            currentLat = finalGeodetic[0]
            currentLon = finalGeodetic[1]
            isSmoothReanchoringActive = false
        }

        trajectoryIntegrator.addPoint(
            TrajectoryPoint(
                latitude = currentLat,
                longitude = currentLon,
                altitude = -ekf.state[2],
                speedMps = getState().speed,
                headingDeg = currentHeadingDeg,
                timestampNs = System.nanoTime(),
                isBlackout = isBlackoutMode,
                isStationary = zuptDetector.isNavStationary
            ),
            isReanchoring = true
        )
        return isSmoothReanchoringActive
    }

    /**
     * Advance one smooth re-anchoring step.
     * Returns true if re-anchoring is still actively transitioning, false when complete.
     */
    fun stepSmoothReanchoring(): Boolean {
        val active: Boolean
        synchronized(stateLock) {
            active = stepSmoothReanchoringLocked()
        }
        dispatchState()
        return active
    }

    fun getState(): NavigationEngineState {
        synchronized(stateLock) {
            val vN = ekf.state[3]
            val vE = ekf.state[4]
            var speed = sqrt(vN * vN + vE * vE).toFloat()
            if (zuptDetector.isNavStationary || speed < 0.1f) {
                speed = 0f
            }

            val naiveGeodetic = transformer.addMetricDisplacementToGeodetic(
                originLat, originLon,
                naiveIntegrator.displacementNorth,
                naiveIntegrator.displacementEast
            )

            // 1-sigma circular position uncertainty from the EKF's own covariance --
            // grows on its own during blackout since no GNSS position update shrinks it.
            val uncertaintyRadius = sqrt(max(0.0, ekf.P[0][0]) + max(0.0, ekf.P[1][1]))

            return NavigationEngineState(
                latitude = currentLat,
                longitude = currentLon,
                altitude = -ekf.state[2],
                speed = speed,
                heading = currentHeadingDeg,
                distanceTravelled = trajectoryIntegrator.getTotalDistance(),
                isInitialized = isEngineInitialized,
                isStationary = zuptDetector.isNavStationary,
                motionState = zuptDetector.motionState.name,
                isBlackout = isBlackoutMode,
                timestampNs = lastStateTimestampNs,
                latestGateAction = currentGateAction.name,
                acceptedCount = acceptedPredictionCount,
                clampedCount = clampedPredictionCount,
                rejectedCount = rejectedPredictionCount,
                naiveLatitude = naiveGeodetic[0],
                naiveLongitude = naiveGeodetic[1],
                uncertaintyRadiusMeters = uncertaintyRadius
            )
        }
    }

    val isStationary: Boolean
        get() = zuptDetector.isNavStationary

    val isInitialized: Boolean
        get() = isEngineInitialized

    val distanceTravelled: Double
        get() = trajectoryIntegrator.getTotalDistance()

    val lastInferenceLatency: Float
        get() = lastInferenceLatencyMs

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
            lastSanitizedGnssSpeed = 0f
            lastGnssSpeedTimestampNs = 0L
            blackoutEntrySpeedMps = 0f
            blackoutEntryTimestampNs = 0L
            lastInferenceTimestampNs = 0L
            lastInferenceLatencyMs = 0f
            lastDisplacementMeters = floatArrayOf(0f, 0f, 0f)
            currentGateAction = GateAction.ACCEPTED
            acceptedPredictionCount = 0
            clampedPredictionCount = 0
            rejectedPredictionCount = 0
            reanchorRemainingN = 0.0
            reanchorRemainingE = 0.0
            isSmoothReanchoringActive = false
            diagnosticRecorder.clear()
            imuBuffer.reset()
            zuptDetector.reset()
            ekf.reset()
            trajectoryIntegrator.reset()
            naiveIntegrator.reset()
        }
    }

    override fun close() {
        modelRunner.close()
    }
}
