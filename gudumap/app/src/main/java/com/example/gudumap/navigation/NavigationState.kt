package com.example.gudumap.navigation

/**
 * Immutable navigation state emitted by the navigation engine to the ViewModel and UI.
 */
data class NavigationState(
    val latitude: Double = 0.0, // 0.0/0.0 is a "no real fix yet" sentinel -- see hasGpsFix
    val longitude: Double = 0.0,
    val speedKmh: Float = 0f,
    val headingDeg: Float = 0f,
    val distanceMeters: Double = 0.0,
    val gnssStatus: String = "UNAVAILABLE", // "AVAILABLE" / "UNAVAILABLE"
    val navigationMode: String = "GNSS",     // "GNSS" / "DEAD RECKONING"
    val mlStatus: String = "INACTIVE",       // "ACTIVE" / "INACTIVE" / "ERROR"
    val ekfStatus: String = "ACTIVE",        // "ACTIVE" / "INACTIVE"
    val mapStatus: String = "OFFLINE",       // "ONLINE" / "OFFLINE"
    val offlineMapStatus: String = "AVAILABLE", // "AVAILABLE" / "LOADING" / "ERROR" / "NOT_AVAILABLE"
    val positionErrorMeters: Double = 0.0,
    val driftPercentage: Double = 0.0,
    val mlInferenceLatencyMs: Long = 0L,
    val accelerometerActive: Boolean = false,
    val gyroscopeActive: Boolean = false,
    val magnetometerActive: Boolean = false,
    val blackoutMode: Boolean = false,
    val gnssRecovered: Boolean = false,
    val recoveryDriftMeters: Double = 0.0,
    val recoveryErrorPercent: Double = 0.0,
    val currentRoadName: String = "",
    val motionState: String = "STATIONARY", // "STATIONARY" / "MOVING" / "ROTATING_IN_PLACE"
    val gnssNavigationMode: String = "GNSS_AVAILABLE", // "GNSS_AVAILABLE" / "GNSS_BLACKOUT" / "GNSS_RECOVERY"
    val blackoutMetrics: BlackoutMetrics = BlackoutMetrics(),
    val latestGateAction: String = "ACCEPTED", // "ACCEPTED" / "CLAMPED" / "REJECTED"
    val acceptedCount: Int = 0,
    val clampedCount: Int = 0,
    val rejectedCount: Int = 0,
    val gnssGroundTruthLat: Double? = null,
    val gnssGroundTruthLon: Double? = null,
    val blackoutDurationSeconds: Double = 0.0,
    val naiveLatitude: Double = 0.0, // Uncorrected dead-reckoning position, for demo contrast only
    val naiveLongitude: Double = 0.0,
    val uncertaintyRadiusMeters: Double = 0.0, // EKF 1-sigma circular position uncertainty
    val headingConfidence: String = "UNRELIABLE", // "HIGH" / "MEDIUM" / "LOW" / "UNRELIABLE" -- Android's own magnetometer/rotation-vector reliability signal
    val motionMode: String = "VEHICLE_MODE", // "VEHICLE_MODE" / "CONSERVATIVE_MODE" -- pedestrian-safe fallback classification, decided once at blackout entry (PROJECT_STATUS.md §24); UI hookup pending
    val hasGpsFix: Boolean = false, // true once at least one real GNSS fix has ever been obtained this session
    val isInternetAvailable: Boolean = true, // true when device has active internet, false when offline (merged back from teammate's branch, PROJECT_STATUS.md §26; not yet wired to any producer on this branch -- see NavigationEngine.kt merge decision)
    val timestampNs: Long = System.nanoTime()
)
