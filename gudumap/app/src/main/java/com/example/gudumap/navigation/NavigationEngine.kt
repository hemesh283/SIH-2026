package com.example.gudumap.navigation

import android.content.Context
import android.location.Location
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.util.Log
import com.example.gudumap.map.MapMatcher
import com.example.gudumap.map.OfflineMapManager
import com.example.gudumap.sensor.ImuSample
import com.example.gudumap.sensor.OrientationSample
import com.example.gudumap.sensors.LocationManager
import com.example.gudumap.sensors.SensorFusionManager
import com.example.gudumap.sensors.SensorManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * Main Navigation Engine integrating:
 * 1. Sensor Management & Hardware Fusion
 * 2. 10 Hz Native IO-VNBD GRU Model (Local Edge ML Inference)
 * 3. Production Dead Reckoning & IMU Resampling Buffer (10 Hz, 20x6, stride 10)
 * 4. 6-State Extended Kalman Filter (EKF) with Joseph-form stability
 * 5. Multi-signal Zero-Velocity Updates (ZUPT) and Non-Holonomic Constraints (NHC)
 * 6. Offline Coimbatore Map Matching & Offline Map Manager
 * 7. GNSS Blackout & Recovery Modes
 * 8. Automatic Internet-Loss-Triggered Blackout & Lifecycle Pause/Resume
 */
class NavigationEngine(
    private val context: Context,
    private val sensorManager: SensorManager = SensorManager(context),
    private val sensorFusionManager: SensorFusionManager = SensorFusionManager(),
    private val locationManager: LocationManager = LocationManager(context),
    val deadReckoningEngine: DeadReckoningEngine = DeadReckoningEngine(),
    private val mapMatcher: MapMatcher = MapMatcher(context),
    private val offlineMapManager: OfflineMapManager = OfflineMapManager(context)
) {

    companion object {
        private const val TAG = "Gudumap:NavEngine"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _state = MutableStateFlow(NavigationState())
    val state: StateFlow<NavigationState> = _state.asStateFlow()

    private var gnssNavMode = "GNSS_AVAILABLE"
    private var blackoutActive = false
    private var latestRawGnssLocation: Location? = null

    // Automatic internet-loss-triggered blackout (merged back from teammate's branch,
    // PROJECT_STATUS.md §27). isInternetAvailable seeds from OfflineMapManager.isOnline() (the
    // stricter, NET_CAPABILITY_VALIDATED-checking version -- PROJECT_STATUS.md §26 Q1), and the
    // NetworkCallback's own request below also requires NET_CAPABILITY_VALIDATED, so this
    // feature stays consistent with that decision end to end, not just at startup.
    private val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
    private var isInternetAvailable = true
    private var autoTriggeredByNetworkLoss = false
    private var networkCallback: ConnectivityManager.NetworkCallback? = null

    // Blackout tracking fields
    private var blackoutStartTimeMs = 0L
    private var blackoutEndTimeMs = 0L
    private var blackoutStartLat = 11.0168
    private var blackoutStartLon = 76.9558
    private var blackoutStartHeading = 0f
    private var blackoutStartSpeed = 0f
    private var blackoutStartDist = 0.0

    private var blackoutStationaryDurationSec = 0.0
    private var blackoutRotatingDurationSec = 0.0
    private var blackoutMaxErrorMeters = 0.0
    private var blackoutMaxSpeedKmh = 0f
    private var blackoutSpeedSum = 0.0
    private var blackoutSpeedCount = 0
    private var blackoutAcceptedStart = 0
    private var blackoutClampedStart = 0
    private var blackoutRejectedStart = 0

    private var storedBlackoutMetrics = BlackoutMetrics()
    private var lastStateEmitTimeMs = 0L

    init {
        // Deliberately do NOT initialize DeadReckoningEngine with any default position here
        // (Fix 2, PROJECT_STATUS.md §11 -- previously hardcoded to Coimbatore, which silently
        // anchored the whole session there if blackout was entered before a real GPS fix
        // arrived). The engine stays uninitialized (isInitialized == false) until either a real
        // GNSS fix arrives via correctWithGnss(), or setBlackoutMode(true) is explicitly refused
        // below because no fix exists yet.
        deadReckoningEngine.mapMatcher = OsmRoadNetworkMapMatcher(mapMatcher)
        deadReckoningEngine.initializeAssets(context)
    }

    fun start() {
        Log.i(TAG, "Starting Gudumap Navigation Engine...")
        deadReckoningEngine.initializeAssets(context)
        startSensors()
        startLocationListening()
        registerNetworkCallback()
    }

    /**
     * Registers all four motion sensors. Extracted from [start] (merged back from teammate's
     * branch, PROJECT_STATUS.md §27) so [resume] can re-register them after [pause] stopped
     * them, without duplicating this wiring.
     */
    private fun startSensors() {
        // 1. Rotation Vector
        sensorManager.startRotationVector { data ->
            sensorFusionManager.updateRotationVector(data.rotationMatrix, data.timestampNs)
            val orientation = sensorFusionManager.getOrientation()
            deadReckoningEngine.updateOrientation(
                OrientationSample(
                    timestampNs = data.timestampNs,
                    rotationMatrix = data.rotationMatrix,
                    quaternion = floatArrayOf(0f, 0f, 0f, 1f),
                    azimuthRad = Math.toRadians(orientation.heading.toDouble()).toFloat(),
                    pitchRad = Math.toRadians(orientation.pitch.toDouble()).toFloat(),
                    rollRad = Math.toRadians(orientation.roll.toDouble()).toFloat()
                )
            )
        }

        // 2. Accelerometer
        sensorManager.startAccelerometer { data ->
            sensorFusionManager.updateAccelerometer(data.x, data.y, data.z, data.timestampNs)
            onSensorStep(data.timestampNs, data.x, data.y, data.z, null, null, null)
        }

        // 3. Gyroscope
        sensorManager.startGyroscope { data ->
            sensorFusionManager.updateGyroscope(data.x, data.y, data.z, data.timestampNs)
            onSensorStep(data.timestampNs, null, null, null, data.x, data.y, data.z)
        }

        // 4. Magnetometer
        sensorManager.startMagnetometer { data ->
            sensorFusionManager.updateMagnetometer(data.x, data.y, data.z, data.timestampNs)
        }
    }

    /**
     * Stops all sensors and location updates (merged back from teammate's branch,
     * PROJECT_STATUS.md §27) -- call when the app backgrounds, so it isn't running a full
     * sensor+location pipeline indefinitely while nobody is looking at it. Does not touch
     * [networkCallback] -- connectivity monitoring for the auto-blackout feature keeps running
     * regardless of pause state, since losing internet is meaningful whether or not the UI is
     * currently visible.
     */
    fun pause() {
        Log.i(TAG, "Pausing Gudumap Navigation Engine (unregistering sensors & location listener for background safety)")
        sensorManager.stopAll()
        locationManager.stopLocationUpdates()
    }

    /**
     * Re-registers sensors and location updates after [pause] (PROJECT_STATUS.md §27). Reuses
     * [retryLocationUpdatesIfNeeded] rather than calling [startLocationListening] directly, so
     * this stays safe even if permission was revoked while backgrounded.
     */
    fun resume() {
        Log.i(TAG, "Resuming Gudumap Navigation Engine (re-registering sensors)")
        startSensors()
        retryLocationUpdatesIfNeeded()
    }

    /**
     * Automatic internet-loss-triggered blackout (merged back from teammate's branch,
     * PROJECT_STATUS.md §27). Until now the ONLY way to enter blackout/dead-reckoning mode was
     * the manual on-screen button; this watches Android's own connectivity state and enters
     * blackout automatically the moment real internet is genuinely lost (as long as a real GPS
     * fix already exists to anchor from -- never falls back to any default position, same
     * guarantee [setBlackoutMode] already enforces for the manual path).
     *
     * The network request requires NET_CAPABILITY_VALIDATED, not just NET_CAPABILITY_INTERNET
     * -- matching the stricter isOnline() decision (PROJECT_STATUS.md §26 Q1). Without this, a
     * captive-portal Wi-Fi (common at demo/conference venues) that LOOKS connected but isn't
     * would fire onAvailable() and could suppress this exact safety feature at the worst
     * possible moment, or end an auto-triggered blackout while genuine internet still isn't
     * actually working.
     *
     * autoTriggeredByNetworkLoss distinguishes an auto-triggered blackout from a manually
     * started one: only a blackout THIS callback started gets automatically ended when internet
     * returns. A blackout a person started deliberately via the button is left alone even if
     * internet happens to come back mid-blackout -- ending it wasn't an automatic decision to
     * make on the user's behalf.
     */
    private fun registerNetworkCallback() {
        isInternetAvailable = offlineMapManager.isOnline()
        try {
            val request = NetworkRequest.Builder()
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .addCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
                .build()

            networkCallback = object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    Log.i(TAG, "Network Callback: Internet is ONLINE")
                    isInternetAvailable = true
                    if (autoTriggeredByNetworkLoss && blackoutActive) {
                        autoTriggeredByNetworkLoss = false
                        setBlackoutMode(false, isAutomatic = true)
                    }
                    emitThrottledState(force = true)
                }

                override fun onLost(network: Network) {
                    Log.i(TAG, "Network Callback: Internet is OFFLINE -- auto-activating AI location prediction")
                    isInternetAvailable = false
                    if (!blackoutActive && latestRawGnssLocation != null) {
                        autoTriggeredByNetworkLoss = true
                        setBlackoutMode(true, isAutomatic = true)
                    }
                    emitThrottledState(force = true)
                }

                override fun onUnavailable() {
                    Log.i(TAG, "Network Callback: Internet is UNAVAILABLE")
                    isInternetAvailable = false
                    if (!blackoutActive && latestRawGnssLocation != null) {
                        autoTriggeredByNetworkLoss = true
                        setBlackoutMode(true, isAutomatic = true)
                    }
                    emitThrottledState(force = true)
                }
            }
            connectivityManager?.registerNetworkCallback(request, networkCallback!!)
        } catch (e: Exception) {
            Log.w(TAG, "Could not register network callback: ${e.message}")
        }
    }

    /**
     * Registers for location updates. Extracted from [start] so it can also be called again
     * later by [retryLocationUpdatesIfNeeded] -- fixes the registration-timing gap where
     * permission granted after [start] already ran was never retried (PROJECT_STATUS.md §13/14).
     */
    private fun startLocationListening() {
        locationManager.startLocationUpdates(
            onLocationChanged = { loc ->
                onGnssLocationChanged(loc)
            },
            onStatusChanged = { available ->
                _state.update { it.copy(gnssStatus = if (available && !blackoutActive) "AVAILABLE" else "UNAVAILABLE") }
            }
        )
    }

    /**
     * Re-registers for location updates if permission is now granted but we aren't already
     * listening -- fixes the case where [start] ran before permission was granted (the user
     * taps "Allow" seconds later) and the case where permission is granted via system Settings
     * while the app is backgrounded, then the app is resumed (PROJECT_STATUS.md §13/14). Safe
     * to call repeatedly (e.g. every onResume); a no-op if already listening or still denied.
     */
    fun retryLocationUpdatesIfNeeded() {
        if (locationManager.isListening()) {
            return
        }
        if (!locationManager.hasLocationPermission()) {
            Log.i(TAG, "retryLocationUpdatesIfNeeded: permission still not granted")
            return
        }
        Log.i(TAG, "retryLocationUpdatesIfNeeded: permission now granted, registering")
        startLocationListening()
    }

    private var lastLinearAccX = 0f
    private var lastLinearAccY = 0f
    private var lastLinearAccZ = 0f
    private var lastGyroX = 0f
    private var lastGyroY = 0f
    private var lastGyroZ = 0f

    @Synchronized
    private fun onSensorStep(
        timestampNs: Long,
        ax: Float?, ay: Float?, az: Float?,
        gx: Float?, gy: Float?, gz: Float?
    ) {
        ax?.let { lastLinearAccX = it }
        ay?.let { lastLinearAccY = it }
        az?.let { lastLinearAccZ = it }
        gx?.let { lastGyroX = it }
        gy?.let { lastGyroY = it }
        gz?.let { lastGyroZ = it }

        val imuSample = ImuSample(
            timestampNs = timestampNs,
            ax = lastLinearAccX,
            ay = lastLinearAccY,
            az = lastLinearAccZ,
            gx = lastGyroX,
            gy = lastGyroY,
            gz = lastGyroZ
        )
        deadReckoningEngine.addSensorSample(imuSample)

        emitThrottledState()
    }

    private fun onGnssLocationChanged(location: Location) {
        // TEMPORARY DIAGNOSTIC LOGGING (PROJECT_STATUS.md §13): this is the exact point
        // hasGpsFix flips true on the next emitThrottledState() tick. If this line never
        // appears in logcat, NavigationEngine itself never received a callback from
        // LocationManager -- check LocationManager's own logs to see why.
        val wasFirstFix = (latestRawGnssLocation == null)
        latestRawGnssLocation = location
        if (wasFirstFix) {
            Log.i(TAG, "onGnssLocationChanged: FIRST real GPS fix received -- " +
                "hasGpsFix will become true, lat=${location.latitude} lon=${location.longitude}")
        }

        when (gnssNavMode) {
            "GNSS_AVAILABLE" -> {
                deadReckoningEngine.correctWithGnss(location)
            }
            "GNSS_BLACKOUT" -> {
                // Blackout Active: Do NOT feed GNSS into DR/EKF.
                Log.i("GUDUMAP_BLACKOUT", String.format(Locale.US, "GNSS_GROUND_TRUTH: lat=%.6f lon=%.6f", location.latitude, location.longitude))

                val drState = deadReckoningEngine.getState()
                val posErr = CoordinateTransformer.computeDistanceBetween(
                    drState.latitude, drState.longitude,
                    location.latitude, location.longitude
                )
                val drDist = max(0.0, drState.distanceTravelled - blackoutStartDist)
                val driftPct = if (drDist > 1.0) (posErr / drDist) * 100.0 else 0.0
                blackoutMaxErrorMeters = max(blackoutMaxErrorMeters, posErr)

                _state.update {
                    it.copy(
                        positionErrorMeters = posErr,
                        driftPercentage = driftPct,
                        gnssGroundTruthLat = location.latitude,
                        gnssGroundTruthLon = location.longitude
                    )
                }
            }
            "GNSS_RECOVERY" -> {
                if (deadReckoningEngine.isReanchoring) {
                    val active = deadReckoningEngine.stepSmoothReanchoring()
                    if (!active) {
                        gnssNavMode = "GNSS_AVAILABLE"
                        deadReckoningEngine.correctWithGnss(location)
                    }
                } else {
                    gnssNavMode = "GNSS_AVAILABLE"
                    deadReckoningEngine.correctWithGnss(location)
                }
            }
        }

        emitThrottledState(force = true)
    }

    /**
     * Explicit API to set GNSS blackout state.
     *
     * @param isAutomatic true only when this call originates from [registerNetworkCallback]'s
     * own handlers (PROJECT_STATUS.md §27/§28) -- every other caller (the manual UI button via
     * [NavigationViewModel], [toggleBlackout]) uses the default `false`. A manual call resets
     * [autoTriggeredByNetworkLoss] to false, so that flag can never outlive the specific
     * auto-triggered blackout session it was set for. Without this, a manually-ended-then-
     * manually-restarted blackout (both while still offline) would inherit a stale `true` from
     * an earlier, unrelated auto-triggered cycle and get incorrectly auto-ended the next time
     * internet returns (§27's disclosed edge case, closed here). Automatic calls skip this
     * reset -- they manage the flag themselves, immediately before/after calling this function.
     */
    fun setBlackoutMode(enabled: Boolean, isAutomatic: Boolean = false) {
        if (enabled == blackoutActive) return

        if (!isAutomatic) {
            autoTriggeredByNetworkLoss = false
        }

        if (enabled) {
            // Refuse to enter blackout without ever having had a real GPS fix (Fix 2,
            // PROJECT_STATUS.md §11) -- previously fell back to whatever drCurrent.latitude
            // happened to be, which was a hardcoded Coimbatore default if no fix had arrived.
            // Never fall back to any default here; the UI (gated on hasGpsFix) is the primary
            // prevention -- this is a defense-in-depth backstop for this method specifically.
            if (latestRawGnssLocation == null) {
                Log.w(TAG, "setBlackoutMode(true) refused: no real GPS fix obtained yet")
                return
            }

            // START GNSS BLACKOUT
            blackoutActive = true
            gnssNavMode = "GNSS_BLACKOUT"

            // Guaranteed non-null past the guard above -- no fallback to any default here.
            val gnssAtEntry = latestRawGnssLocation!!
            blackoutStartLat = gnssAtEntry.latitude
            blackoutStartLon = gnssAtEntry.longitude
            blackoutStartHeading = gnssAtEntry.bearing
            // Route through the same sanitizer correctWithGnss() uses internally (PROJECT_STATUS.md
            // §17) -- this raw Location.getSpeed() value previously seeded the EKF's blackout-entry
            // velocity completely unfiltered, bypassing the Fix from §15 entirely since initialize()
            // never routes through correctWithGnss(). Reuses the same bound logic, doesn't duplicate it.
            blackoutStartSpeed = deadReckoningEngine.sanitizeExternalGnssSpeed(
                gnssAtEntry.speed,
                gnssAtEntry.elapsedRealtimeNanos
            ) ?: 0f
            blackoutStartTimeMs = System.currentTimeMillis()
            blackoutStartDist = deadReckoningEngine.distanceTravelled

            blackoutStationaryDurationSec = 0.0
            blackoutRotatingDurationSec = 0.0
            blackoutMaxErrorMeters = 0.0
            blackoutMaxSpeedKmh = 0f
            blackoutSpeedSum = 0.0
            blackoutSpeedCount = 0
            blackoutAcceptedStart = deadReckoningEngine.acceptedCount
            blackoutClampedStart = deadReckoningEngine.clampedCount
            blackoutRejectedStart = deadReckoningEngine.rejectedCount

            // Initialize / anchor DR reference at start position
            deadReckoningEngine.initialize(
                latitude = blackoutStartLat,
                longitude = blackoutStartLon,
                speedMps = blackoutStartSpeed,
                bearingDeg = blackoutStartHeading
            )
            deadReckoningEngine.setBlackoutMode(true)

            val liveMetrics = BlackoutMetrics(
                blackoutStartTime = blackoutStartTimeMs,
                blackoutStartLatitude = blackoutStartLat,
                blackoutStartLongitude = blackoutStartLon,
                drLatitude = blackoutStartLat,
                drLongitude = blackoutStartLon
            )
            storedBlackoutMetrics = liveMetrics

            Log.i("GUDUMAP_BLACKOUT", String.format(Locale.US,
                "BLACKOUT_START: lat=%.6f lon=%.6f heading=%.1f speed=%.2f",
                blackoutStartLat, blackoutStartLon, blackoutStartHeading, blackoutStartSpeed
            ))

            _state.update {
                it.copy(
                    blackoutMode = true,
                    gnssNavigationMode = "GNSS_BLACKOUT",
                    navigationMode = "DEAD RECKONING",
                    gnssStatus = "UNAVAILABLE",
                    mapStatus = "OFFLINE",
                    gnssRecovered = false,
                    positionErrorMeters = 0.0,
                    driftPercentage = 0.0,
                    blackoutMetrics = liveMetrics,
                    blackoutDurationSeconds = 0.0
                )
            }
        } else {
            // END GNSS BLACKOUT (GNSS RECOVERY)
            blackoutActive = false
            gnssNavMode = "GNSS_RECOVERY"
            deadReckoningEngine.setBlackoutMode(false)

            blackoutEndTimeMs = System.currentTimeMillis()
            val durationSec = max(0.0, (blackoutEndTimeMs - blackoutStartTimeMs) / 1000.0)

            val drState = deadReckoningEngine.getState()
            val drDist = max(0.0, drState.distanceTravelled - blackoutStartDist)

            val recoveryGps = latestRawGnssLocation
            val endGnssLat = recoveryGps?.latitude ?: drState.latitude
            val endGnssLon = recoveryGps?.longitude ?: drState.longitude

            val gnssRefDist = CoordinateTransformer.computeDistanceBetween(
                blackoutStartLat, blackoutStartLon,
                endGnssLat, endGnssLon
            )
            val posError = CoordinateTransformer.computeDistanceBetween(
                drState.latitude, drState.longitude,
                endGnssLat, endGnssLon
            )
            val distError = abs(drDist - gnssRefDist)
            val driftPct = if (drDist > 1.0) (posError / drDist) * 100.0 else 0.0
            val avgSpeed = if (blackoutSpeedCount > 0) (blackoutSpeedSum / blackoutSpeedCount).toFloat() else 0f

            val acceptedInBlackout = max(0, deadReckoningEngine.acceptedCount - blackoutAcceptedStart)
            val clampedInBlackout = max(0, deadReckoningEngine.clampedCount - blackoutClampedStart)
            val rejectedInBlackout = max(0, deadReckoningEngine.rejectedCount - blackoutRejectedStart)

            val finalMetrics = BlackoutMetrics(
                blackoutStartTime = blackoutStartTimeMs,
                blackoutEndTime = blackoutEndTimeMs,
                blackoutDurationSeconds = durationSec,
                blackoutStartLatitude = blackoutStartLat,
                blackoutStartLongitude = blackoutStartLon,
                blackoutEndGnssLatitude = endGnssLat,
                blackoutEndGnssLongitude = endGnssLon,
                drLatitude = drState.latitude,
                drLongitude = drState.longitude,
                gnssReferenceDistance = gnssRefDist,
                drDistance = drDist,
                positionErrorMeters = posError,
                distanceErrorMeters = distError,
                driftPercentage = driftPct,
                maximumPositionErrorMeters = max(blackoutMaxErrorMeters, posError),
                maximumSpeedKmh = blackoutMaxSpeedKmh,
                averageSpeedKmh = avgSpeed,
                mlInferenceMs = deadReckoningEngine.lastInferenceLatency.toLong(),
                numberOfAcceptedMLPredictions = acceptedInBlackout,
                numberOfClampedMLPredictions = clampedInBlackout,
                numberOfRejectedMLPredictions = rejectedInBlackout,
                stationaryDuration = blackoutStationaryDurationSec,
                rotatingInPlaceDuration = blackoutRotatingDurationSec
            )
            storedBlackoutMetrics = finalMetrics

            Log.i("GUDUMAP_BLACKOUT", String.format(Locale.US,
                "BLACKOUT_RECOVERY: duration=%.1fs drDistance=%.1fm gnssDistance=%.1fm positionError=%.1fm driftPercent=%.1f%%",
                durationSec, drDist, gnssRefDist, posError, driftPct
            ))

            // Smooth re-anchoring transition
            if (recoveryGps != null) {
                deadReckoningEngine.startSmoothReanchoring(endGnssLat, endGnssLon)
            } else {
                gnssNavMode = "GNSS_AVAILABLE"
            }

            _state.update {
                it.copy(
                    blackoutMode = false,
                    gnssNavigationMode = gnssNavMode,
                    navigationMode = "GNSS",
                    mapStatus = offlineMapManager.getMapStatus(false),
                    gnssRecovered = true,
                    recoveryDriftMeters = posError,
                    recoveryErrorPercent = driftPct,
                    positionErrorMeters = posError,
                    driftPercentage = driftPct,
                    blackoutMetrics = finalMetrics,
                    blackoutDurationSeconds = durationSec
                )
            }
        }

        emitThrottledState(force = true)
    }

    fun toggleBlackout() {
        setBlackoutMode(!blackoutActive)
    }

    private fun emitThrottledState(force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && now - lastStateEmitTimeMs < 80) return // ~12 fps state update for Compose
        val dtSec = if (lastStateEmitTimeMs > 0L) (now - lastStateEmitTimeMs) / 1000.0 else 0.08
        lastStateEmitTimeMs = now

        val drState = deadReckoningEngine.getState()
        val isStationary = drState.isStationary
        val motionStateStr = drState.motionState

        var currentLat = drState.latitude
        var currentLon = drState.longitude
        var currentSpeedKmh = if (isStationary) 0f else drState.speed * 3.6f
        if (currentSpeedKmh < 0.2f) currentSpeedKmh = 0f
        val currentHeadingDeg = drState.heading

        // Road-level map matching against Coimbatore OSM
        val matchResult = mapMatcher.match(currentLat, currentLon, currentHeadingDeg, currentSpeedKmh)
        if (matchResult.isSnapped) {
            currentLat = matchResult.matchedLat
            currentLon = matchResult.matchedLon
        }

        val mlStatusStr = if (deadReckoningEngine.modelRunner.ready) "ACTIVE" else "INACTIVE"

        // Forward Android's own magnetometer/rotation-vector reliability signal -- previously
        // discarded in SensorManager.onAccuracyChanged -- so heading confidence is visible
        // instead of assumed constant, especially relevant inside a vehicle/tunnel blackout.
        sensorFusionManager.updateMagnetometerAccuracy(sensorManager.magnetometerAccuracy)
        sensorFusionManager.updateRotationVectorAccuracy(sensorManager.rotationVectorAccuracy)
        val headingConfidenceStr = sensorFusionManager.headingConfidence.name

        // Step smooth re-anchoring transition if in recovery
        if (gnssNavMode == "GNSS_RECOVERY") {
            val stillActive = deadReckoningEngine.stepSmoothReanchoring()
            if (!stillActive) {
                gnssNavMode = "GNSS_AVAILABLE"
            }
        }

        val liveMetrics = if (blackoutActive) {
            val elapsedSec = max(0.0, (now - blackoutStartTimeMs) / 1000.0)
            val drDist = max(0.0, drState.distanceTravelled - blackoutStartDist)
            if (motionStateStr == "STATIONARY") blackoutStationaryDurationSec += dtSec
            if (motionStateStr == "ROTATING_IN_PLACE") blackoutRotatingDurationSec += dtSec
            blackoutMaxSpeedKmh = max(blackoutMaxSpeedKmh, currentSpeedKmh)
            blackoutSpeedSum += currentSpeedKmh
            blackoutSpeedCount++

            val acceptedInBlackout = max(0, drState.acceptedCount - blackoutAcceptedStart)
            val clampedInBlackout = max(0, drState.clampedCount - blackoutClampedStart)
            val rejectedInBlackout = max(0, drState.rejectedCount - blackoutRejectedStart)

            val currentError = _state.value.positionErrorMeters

            Log.i("GUDUMAP_BLACKOUT", String.format(Locale.US,
                "DR_UPDATE: t=%d lat=%.6f lon=%.6f speed=%.2f distance=%.2f motionState=%s mlGate=%s motionMode=%s",
                now, currentLat, currentLon, currentSpeedKmh, drDist, motionStateStr, drState.latestGateAction, drState.motionMode
            ))

            BlackoutMetrics(
                blackoutStartTime = blackoutStartTimeMs,
                blackoutDurationSeconds = elapsedSec,
                blackoutStartLatitude = blackoutStartLat,
                blackoutStartLongitude = blackoutStartLon,
                drLatitude = currentLat,
                drLongitude = currentLon,
                drDistance = drDist,
                positionErrorMeters = currentError,
                driftPercentage = _state.value.driftPercentage,
                maximumPositionErrorMeters = blackoutMaxErrorMeters,
                maximumSpeedKmh = blackoutMaxSpeedKmh,
                averageSpeedKmh = if (blackoutSpeedCount > 0) (blackoutSpeedSum / blackoutSpeedCount).toFloat() else 0f,
                mlInferenceMs = deadReckoningEngine.lastInferenceLatency.toLong(),
                numberOfAcceptedMLPredictions = acceptedInBlackout,
                numberOfClampedMLPredictions = clampedInBlackout,
                numberOfRejectedMLPredictions = rejectedInBlackout,
                stationaryDuration = blackoutStationaryDurationSec,
                rotatingInPlaceDuration = blackoutRotatingDurationSec
            )
        } else {
            storedBlackoutMetrics
        }

        _state.update {
            it.copy(
                latitude = currentLat,
                longitude = currentLon,
                speedKmh = currentSpeedKmh,
                headingDeg = currentHeadingDeg,
                distanceMeters = drState.distanceTravelled,
                gnssStatus = if (!blackoutActive && locationManager.isGnssAvailable) "AVAILABLE" else "UNAVAILABLE",
                navigationMode = if (blackoutActive) "DEAD RECKONING" else "GNSS",
                gnssNavigationMode = gnssNavMode,
                mlStatus = mlStatusStr,
                ekfStatus = if (deadReckoningEngine.ekf.isInitialized) "ACTIVE" else "INACTIVE",
                mapStatus = "OFFLINE",
                offlineMapStatus = offlineMapManager.getOfflineMapStatusString(),
                mlInferenceLatencyMs = deadReckoningEngine.lastInferenceLatency.toLong(),
                accelerometerActive = sensorManager.isAccelerometerActive,
                gyroscopeActive = sensorManager.isGyroscopeActive,
                magnetometerActive = sensorManager.isMagnetometerActive,
                hasGpsFix = (latestRawGnssLocation != null),
                currentRoadName = matchResult.roadName,
                motionState = motionStateStr,
                latestGateAction = drState.latestGateAction,
                acceptedCount = drState.acceptedCount,
                clampedCount = drState.clampedCount,
                rejectedCount = drState.rejectedCount,
                blackoutDurationSeconds = if (blackoutActive) ((now - blackoutStartTimeMs) / 1000.0) else storedBlackoutMetrics.blackoutDurationSeconds,
                blackoutMetrics = liveMetrics,
                naiveLatitude = drState.naiveLatitude,
                naiveLongitude = drState.naiveLongitude,
                uncertaintyRadiusMeters = drState.uncertaintyRadiusMeters,
                headingConfidence = headingConfidenceStr,
                motionMode = drState.motionMode,
                isInternetAvailable = isInternetAvailable,
                timestampNs = System.nanoTime()
            )
        }
    }

    fun stop() {
        networkCallback?.let {
            try { connectivityManager?.unregisterNetworkCallback(it) } catch (_: Exception) {}
        }
        networkCallback = null
        sensorManager.stopAccelerometer()
        sensorManager.stopGyroscope()
        sensorManager.stopMagnetometer()
        sensorManager.stopRotationVector()
        locationManager.stopLocationUpdates()
        deadReckoningEngine.close()
    }
}
