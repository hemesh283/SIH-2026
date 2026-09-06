package com.example.gudumap

import com.example.gudumap.navigation.NavigationState
import com.example.gudumap.viewmodel.NavigationViewModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NavigationViewModelTest {

    @Test
    fun testNavigationStateDefaults() {
        val state = NavigationState()
        // 0.0/0.0 is the deliberate "no real GPS fix yet" sentinel (Fix 2, PROJECT_STATUS.md
        // §11) -- previously hardcoded to Coimbatore, which this test locked in as "correct."
        assertEquals(0.0, state.latitude, 1e-4)
        assertEquals(0.0, state.longitude, 1e-4)
        assertFalse(state.hasGpsFix)
        assertEquals(0f, state.speedKmh, 1e-4f)
        assertEquals(0f, state.headingDeg, 1e-4f)
        assertEquals(0.0, state.distanceMeters, 1e-4)
        assertEquals("UNAVAILABLE", state.gnssStatus)
        assertEquals("GNSS", state.navigationMode)
        assertEquals("INACTIVE", state.mlStatus)
        assertEquals("ACTIVE", state.ekfStatus)
        assertEquals("OFFLINE", state.mapStatus)
        assertEquals("AVAILABLE", state.offlineMapStatus)
        assertEquals(0.0, state.positionErrorMeters, 1e-4)
        assertEquals(0.0, state.driftPercentage, 1e-4)
        assertEquals(0L, state.mlInferenceLatencyMs)
        assertFalse(state.accelerometerActive)
        assertFalse(state.gyroscopeActive)
        assertFalse(state.magnetometerActive)
        assertFalse(state.blackoutMode)
        assertFalse(state.gnssRecovered)
        assertEquals(0.0, state.recoveryDriftMeters, 1e-4)
        assertEquals(0.0, state.recoveryErrorPercent, 1e-4)
        assertEquals("", state.currentRoadName)
        assertEquals("STATIONARY", state.motionState)
    }

    @Test
    fun testNavigationViewModelFactoryNotNull() {
        assertNotNull(NavigationViewModel.Factory)
    }
}
