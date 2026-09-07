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
 * Binary classification of blackout motion domain (PROJECT_STATUS.md §24). The shipped ML
 * model (gru_io_vnbd.onnx) and NHC were both built/validated for VEHICLE motion (IO-VNBD,
 * ~45km/h driving) -- applying either to walking (sway, stop-start, sideways steps) actively
 * corrupts the estimate rather than helping it. Decided ONCE at blackout entry (mirrors
 * blackoutEntrySpeedMps's existing "captured once, fixed for the whole blackout" pattern) --
 * deliberately not re-evaluated mid-blackout, so behavior stays predictable and auditable.
 */
enum class MotionMode {
    VEHICLE_MODE,
    CONSERVATIVE_MODE
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
    val uncertaintyRadiusMeters: Double = 0.0,
    val motionMode: String = "VEHICLE_MODE" // "VEHICLE_MODE" / "CONSERVATIVE_MODE" -- see MotionMode, PROJECT_STATUS.md §24
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

        // Pedestrian-safe fallback classification (PROJECT_STATUS.md §24).
        //
        // PEDESTRIAN_SPEED_CEILING_MPS = 2.5 m/s (~9 km/h): comfortably above a brisk walking
        // pace (~1.8-2.0 m/s) so genuine walking never grazes it, and comfortably below any
        // speed a vehicle sustains while actually driving (as opposed to stopped) -- a car
        // essentially never cruises this slowly except in a dead-stopped jam or a parking
        // maneuver, both of which are themselves near-zero, not "slow but moving".
        //
        // Why a single instantaneous speed at the moment blackout starts is NOT enough on its
        // own: a vehicle entering blackout (e.g. a tunnel mouth) while briefly stopped at a red
        // light would read ~0 m/s at that exact instant -- indistinguishable from a pedestrian
        // by a single sample. VEHICLE_SPEED_LOOKBACK_SEC = 10.0s fixes this: classification uses
        // the MAX sanitized GNSS speed observed over the preceding 10 seconds, not just the
        // last sample. A car stopped at a light was cruising at normal traffic speed only
        // moments before (traffic lights are commonly tens of seconds apart, and a vehicle's
        // approach/deceleration into a stop is itself visible within a 10s lookback in the
        // overwhelming majority of cases) -- so its rolling-max speed over that window still
        // clears the ceiling even at the instant it happens to be stopped. A pedestrian's
        // rolling-max speed over any 10s window stays below the ceiling by construction, since
        // walking never touches vehicle speeds even briefly. Residual, acknowledged limitation:
        // a vehicle that has been fully stopped (traffic jam, long red light) for the ENTIRE
        // preceding 10+ seconds would still misclassify as CONSERVATIVE_MODE -- accepted as a
        // rare edge case whose failure mode is merely "degraded-but-safe" (this fallback's own
        // intentionally conservative behavior), not the catastrophic drift being fixed here.
        private const val PEDESTRIAN_SPEED_CEILING_MPS = 2.5f
        private const val VEHICLE_SPEED_LOOKBACK_SEC = 10.0

        // Pedestrian-safe fallback, bounded-displacement refinement (PROJECT_STATUS.md §25).
        // The previous session's CONSERVATIVE_MODE fed exactly [0,0,0] displacement every
        // window -- safe, but a frozen 0.0m DR Distance during an indoor/campus walking demo
        // looks like the app has stopped tracking entirely. This lets a small, memoryless,
        // per-window raw displacement through instead (see integrateRawPedestrianDisplacement),
        // hard-capped here so the "cannot reproduce the 2721m runaway" guarantee still holds
        // regardless of what the raw integration happens to suggest.
        //
        // PEDESTRIAN_MAX_SPEED_MPS = 2.0 m/s (~7.2 km/h): average adult walking pace is
        // ~1.4 m/s (~5 km/h); a brisk walk is commonly ~1.8-2.0 m/s. 2.0 m/s sits at the top of
        // that brisk-walking range -- generous enough that a presenter walking normally during
        // a live demo is never artificially clipped below their real pace, while staying
        // clearly under jogging (~2.5+ m/s) and vastly under any vehicle speed. This is a
        // safety CEILING, not an accuracy model: its job is to bound the worst case for every
        // single window, not to estimate the typical one -- applied as a hard clamp on the
        // OUTPUT of the raw integration below, after the fact, never as a trust-the-math input.
        private const val PEDESTRIAN_MAX_SPEED_MPS = 2.0f
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

    // Pedestrian-safe fallback (PROJECT_STATUS.md §24). Rolling history of sanitized GNSS
    // speed readings (timestampNs to speedMps), fed only while GNSS is available, used to
    // classify VEHICLE_MODE vs CONSERVATIVE_MODE once at blackout entry -- see
    // PEDESTRIAN_SPEED_CEILING_MPS/VEHICLE_SPEED_LOOKBACK_SEC for the reasoning. Independent of
    // the EKF/ML pipeline entirely (raw sanitized GNSS only), so it can't be corrupted by
    // exactly the failure mode it exists to detect.
    private val recentGnssSpeedHistory = ArrayDeque<Pair<Long, Float>>()
    private var currentMotionMode: MotionMode = MotionMode.VEHICLE_MODE
    val motionMode: MotionMode get() = synchronized(stateLock) { currentMotionMode }

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
        if (speed != null) {
            recordGnssSpeedForModeClassification(speed, location.elapsedRealtimeNanos)
        }
        correctWithGnss(location.latitude, location.longitude, speed, bearing)
    }

    /**
     * Feeds the rolling speed-history buffer used to classify VEHICLE_MODE vs
     * CONSERVATIVE_MODE at blackout entry (PROJECT_STATUS.md §24). Only called with real,
     * already-sanitized GNSS speed -- independent of the EKF/ML pipeline this classification
     * gates, so it can't be corrupted by the failure mode it exists to detect.
     */
    private fun recordGnssSpeedForModeClassification(speedMps: Float, timestampNs: Long) {
        synchronized(stateLock) {
            recentGnssSpeedHistory.addLast(timestampNs to speedMps)
            val cutoffNs = timestampNs - (VEHICLE_SPEED_LOOKBACK_SEC * 1_000_000_000.0).toLong()
            while (recentGnssSpeedHistory.isNotEmpty() && recentGnssSpeedHistory.first().first < cutoffNs) {
                recentGnssSpeedHistory.removeFirst()
            }
        }
    }

    /**
     * VEHICLE_MODE vs CONSERVATIVE_MODE, decided from the rolling GNSS speed history -- see
     * PEDESTRIAN_SPEED_CEILING_MPS/VEHICLE_SPEED_LOOKBACK_SEC above for the full reasoning. An
     * empty history (no real GNSS speed observed in the lookback window) defaults to
     * CONSERVATIVE_MODE -- with no evidence either way, the safe/degraded fallback is the
     * correct default, not an assumption of vehicle motion.
     */
    private fun classifyMotionMode(): MotionMode {
        val maxRecentSpeedMps = recentGnssSpeedHistory.maxOfOrNull { it.second } ?: 0f
        return if (maxRecentSpeedMps < PEDESTRIAN_SPEED_CEILING_MPS) {
            MotionMode.CONSERVATIVE_MODE
        } else {
            MotionMode.VEHICLE_MODE
        }
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
     * Raw, uncorrected, memoryless per-window displacement estimate for the pedestrian-safe
     * fallback (PROJECT_STATUS.md §25) -- no ML model, no NHC, just a plain double-integration
     * of this window's own accelerometer samples in the vehicle/local frame.
     *
     * Integrates only the LAST `imuBuffer.stride` rows of the window, not all `windowSize` rows.
     * `IMUBuffer` emits overlapping windows (windowSize=20 samples / 2.0s, stride=10 samples /
     * 1.0s) -- each new window's first `stride` rows are literally the same samples as the
     * previous window's last `stride` rows. Integrating the full window every call would count
     * that shared 1.0s of real motion twice across two consecutive windows; integrating only the
     * newest `stride` rows (the 1.0s of data that's actually new since the last call) matches
     * the same dt already used everywhere else in this function (ModelMetadata.STRIDE_DURATION_SEC).
     *
     * Deliberately resets to v=0 at the start of EVERY call -- no velocity is carried between
     * windows (unlike NaiveIntegrator, which intentionally keeps running velocity to show how
     * badly naive dead reckoning drifts on its own -- exactly the failure mode this function
     * must NOT reproduce). A biased or noisy window can only ever affect that one window's own
     * output; there is no persistent state for a per-window error to compound into over time.
     * The caller clamps this raw result to PEDESTRIAN_MAX_SPEED_MPS -- this function does not
     * clamp anything itself, by design (Task 1.3: the cap must apply to the output, not be
     * baked into the estimate).
     */
    private fun integrateRawPedestrianDisplacement(window: Array<FloatArray>): FloatArray {
        val sampleDtSec = (imuBuffer.targetDtNs / 1_000_000_000.0).toFloat()
        val newSamplesStart = max(0, window.size - imuBuffer.stride)

        var vx = 0f
        var vy = 0f
        var dx = 0f
        var dy = 0f
        for (i in newSamplesStart until window.size) {
            val ax = window[i][0] * ModelMetadata.GRAVITY_MPS2
            val ay = window[i][1] * ModelMetadata.GRAVITY_MPS2
            vx += ax * sampleDtSec
            vy += ay * sampleDtSec
            dx += vx * sampleDtSec
            dy += vy * sampleDtSec
        }

        return floatArrayOf(dx, dy, 0f)
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

        // Computed once, used identically in both the gate section below and the EKF-update
        // section further down, so both agree on whether pedestrian fallback is active this
        // window without re-deriving it from gateAction/localDisplacement's contents.
        val isPedestrianFallbackActive = isBlackoutMode && currentMotionMode == MotionMode.CONSERVATIVE_MODE

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
        } else if (isPedestrianFallbackActive) {
            // Pedestrian-safe fallback (PROJECT_STATUS.md §24, refined §25): the shipped ML
            // model and NHC are both validated for vehicle motion only, so neither is invoked
            // here -- no model output of any kind ever reaches the EKF in this mode. Unlike
            // §24's original version (which fed exactly [0,0,0]), this now allows a small,
            // memoryless, per-window raw displacement through (see
            // integrateRawPedestrianDisplacement), hard-capped to PEDESTRIAN_MAX_SPEED_MPS
            // immediately after integration -- a walking-speed-derived ceiling applied to the
            // OUTPUT, not a trust-the-math input, so "cannot reproduce the 2721m runaway" still
            // holds even though displacement is no longer strictly zero. gateAction is still
            // recorded as REJECTED for telemetry (the ML model itself was genuinely never
            // invoked/accepted) -- the EKF-update section below uses isPedestrianFallbackActive
            // directly, not gateAction, to decide whether to apply this displacement and skip NHC.
            val rawPedestrianDisplacement = integrateRawPedestrianDisplacement(window)
            rawMag = sqrt(
                rawPedestrianDisplacement[0] * rawPedestrianDisplacement[0] +
                    rawPedestrianDisplacement[1] * rawPedestrianDisplacement[1]
            )
            val pedestrianCapDist = PEDESTRIAN_MAX_SPEED_MPS * dtF
            maxPlausibleDist = pedestrianCapDist
            gateAction = GateAction.REJECTED
            localDisplacement = if (rawMag > pedestrianCapDist && rawMag > 0.0001f) {
                val scale = pedestrianCapDist / rawMag
                floatArrayOf(rawPedestrianDisplacement[0] * scale, rawPedestrianDisplacement[1] * scale, 0f)
            } else {
                rawPedestrianDisplacement
            }
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

            // 3. EKF prediction step. Uses isPedestrianFallbackActive directly (not gateAction)
            // to decide whether to apply displacement here -- §25's pedestrian fallback also
            // sets gateAction=REJECTED (for ML telemetry purposes), but unlike a genuine
            // vehicle-mode rejection it DOES carry a real (capped) displacement that must reach
            // the EKF, just without NHC. isNavStationary is checked first and wins regardless of
            // mode (Task 1.4): genuinely not moving still means exactly 0.0m, never
            // walking-speed noise.
            if (isNavStationary || (gateAction == GateAction.REJECTED && !isPedestrianFallbackActive)) {
                // When genuinely rejected or stationary, feed [0,0,0] to EKF and enforce zero-velocity
                ekf.predict(doubleArrayOf(0.0, 0.0, 0.0), dt)
                ekf.updateZupt()
                ekf.state[3] = 0.0
                ekf.state[4] = 0.0
                ekf.state[5] = 0.0
            } else {
                // 3b. Coordinate transformation: local displacement -> world NED. Direction
                // handling for the pedestrian fallback (Task 2, §25): applied along whatever
                // heading is currently available, the same as every other displacement source
                // here -- deliberately NOT additionally scaled down when HeadingConfidence is
                // UNRELIABLE. Reasoning: the magnitude cap above already guarantees safety
                // (bounded per window) regardless of whether the direction is correct, so a
                // heading-based scale-down would only trade a small accuracy gain for LESS
                // visible movement -- and indoor venues (this project's actual demo setting)
                // are exactly where magnetometer/rotation-vector confidence is most often
                // degraded, so that trade would work directly against this task's own goal of
                // visible walking-pace tracking indoors. Direction may be wrong; magnitude is
                // always bounded either way.
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

                // 4. Non-Holonomic Constraints (NHC) -- a vehicle-only, forward-only-motion
                // assumption. Must stay off for the pedestrian fallback even now that it
                // applies a non-zero displacement (PROJECT_STATUS.md §24/§25).
                if (nhc.isEnabled && !isPedestrianFallbackActive) {
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
                    "ML_GATE: predicted=%.2fm, max=%.2fm, action=%s, state=%s, mode=%s | |a_h|=%.2f |w|=%.2f | EKF_V_before=%.2f EKF_V_after=%.2f | dDist=%.2f dist=%.2f",
                    rawMag, maxPlausibleDist, gateAction.name, currentMotionState.name, currentMotionMode.name,
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

                // Pedestrian-safe fallback (PROJECT_STATUS.md §24): classify once, at entry,
                // from the rolling pre-blackout GNSS speed history -- see
                // PEDESTRIAN_SPEED_CEILING_MPS/VEHICLE_SPEED_LOOKBACK_SEC for why a lookback
                // window is used instead of the single instantaneous speed. Not re-evaluated
                // for the rest of this blackout.
                currentMotionMode = classifyMotionMode()

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
                currentMotionMode = MotionMode.VEHICLE_MODE // neutral default, only meaningful during blackout
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
                uncertaintyRadiusMeters = uncertaintyRadius,
                motionMode = currentMotionMode.name
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
            recentGnssSpeedHistory.clear()
            currentMotionMode = MotionMode.VEHICLE_MODE
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
