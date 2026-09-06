package com.example.gudumap.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.gudumap.navigation.NavigationEngine
import com.example.gudumap.navigation.NavigationState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel exposing navigation state to Compose UI.
 * Connects directly to the existing NavigationEngine and pipeline:
 * Sensors -> SensorFusion -> ML -> Dead Reckoning -> EKF -> Map Matching -> NavigationEngine -> NavigationState -> ViewModel -> UI
 */
class NavigationViewModel(
    application: Application,
    private val navigationEngine: NavigationEngine = NavigationEngine(application.applicationContext)
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(NavigationState())
    val state: StateFlow<NavigationState> = _state.asStateFlow()

    // Direct property accessors for all required navigation & sensor metrics
    val blackoutMode: Boolean get() = state.value.blackoutMode
    val navigationMode: String get() = state.value.navigationMode
    val gnssRecovered: Boolean get() = state.value.gnssRecovered
    val recoveryDriftMeters: Double get() = state.value.recoveryDriftMeters
    val recoveryErrorPercent: Double get() = state.value.recoveryErrorPercent
    val gnssStatus: String get() = state.value.gnssStatus
    val mlStatus: String get() = state.value.mlStatus
    val ekfStatus: String get() = state.value.ekfStatus
    val mapStatus: String get() = state.value.mapStatus
    val offlineMapStatus: String get() = state.value.offlineMapStatus
    val currentRoadName: String get() = state.value.currentRoadName
    val latitude: Double get() = state.value.latitude
    val longitude: Double get() = state.value.longitude
    val speedKmh: Float get() = state.value.speedKmh
    val headingDeg: Float get() = state.value.headingDeg
    val distanceMeters: Double get() = state.value.distanceMeters
    val positionErrorMeters: Double get() = state.value.positionErrorMeters
    val driftPercentage: Double get() = state.value.driftPercentage
    val mlInferenceLatencyMs: Long get() = state.value.mlInferenceLatencyMs
    val accelerometerActive: Boolean get() = state.value.accelerometerActive
    val gyroscopeActive: Boolean get() = state.value.gyroscopeActive
    val magnetometerActive: Boolean get() = state.value.magnetometerActive
    val motionState: String get() = state.value.motionState
    val gnssNavigationMode: String get() = state.value.gnssNavigationMode
    val latestGateAction: String get() = state.value.latestGateAction
    val acceptedCount: Int get() = state.value.acceptedCount
    val clampedCount: Int get() = state.value.clampedCount
    val rejectedCount: Int get() = state.value.rejectedCount
    val blackoutMetrics: com.example.gudumap.navigation.BlackoutMetrics get() = state.value.blackoutMetrics
    val blackoutDurationSeconds: Double get() = state.value.blackoutDurationSeconds
    val gnssGroundTruthLat: Double? get() = state.value.gnssGroundTruthLat
    val gnssGroundTruthLon: Double? get() = state.value.gnssGroundTruthLon

    init {
        // Collect state emissions from NavigationEngine within viewModelScope
        viewModelScope.launch {
            navigationEngine.state.collect { latestState ->
                _state.value = latestState
            }
        }
        navigationEngine.start()
    }

    fun setBlackoutMode(enabled: Boolean) {
        navigationEngine.setBlackoutMode(enabled)
    }

    fun toggleBlackout() {
        navigationEngine.toggleBlackout()
    }

    /**
     * Re-registers for location updates if permission is now granted but we aren't already
     * listening. Call this from a permission-grant callback and from onResume -- fixes the
     * registration-timing gap where a late permission grant was never retried (PROJECT_STATUS.md
     * §13/14).
     */
    fun retryLocationUpdatesIfNeeded() {
        navigationEngine.retryLocationUpdatesIfNeeded()
    }

    override fun onCleared() {
        super.onCleared()
        navigationEngine.stop()
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val application = checkNotNull(this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as? Application) {
                    "Application not found in ViewModelProvider CreationExtras"
                }
                NavigationViewModel(application)
            }
        }
    }
}
