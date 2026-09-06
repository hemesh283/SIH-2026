package com.example.gudumap

import com.example.gudumap.navigation.IMUBuffer
import com.example.gudumap.sensor.ImuSample
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class IMUBufferTest {

    @Test
    fun testDefaultConstructorParameters() {
        val buffer = IMUBuffer()
        assertEquals(20, buffer.windowSize)
        assertEquals(10, buffer.stride)
        assertEquals(100_000_000L, buffer.targetDtNs)
    }

    @Test
    fun testInsufficientWindowRejection() {
        val buffer = IMUBuffer(windowSize = 20, stride = 10, targetDtNs = 100_000_000L)
        var windowEmitted = false

        buffer.onWindowReadyListener = { _, _ ->
            windowEmitted = true
        }

        // Add 15 samples at 10 Hz (100ms interval = 100_000_000 ns)
        var timeNs = 1_000_000_000L
        for (i in 0 until 15) {
            buffer.addSample(
                ImuSample(timeNs, 0.1f, 0.2f, 0.3f, 0.01f, 0.02f, 0.03f)
            )
            timeNs += 100_000_000L
        }

        assertFalse("Buffer must not emit window before reaching 20 samples", windowEmitted)
    }

    @Test
    fun testWindowFormationAndStrideScheduling() {
        val buffer = IMUBuffer(windowSize = 20, stride = 10, targetDtNs = 100_000_000L)
        val emittedWindows = ArrayList<Array<FloatArray>>()

        buffer.onWindowReadyListener = { window, _ ->
            emittedWindows.add(window)
        }

        // Feed 45 samples at 10 Hz (100ms steps)
        var timeNs = 1_000_000_000L
        for (i in 0 until 45) {
            val sample = ImuSample(
                timestampNs = timeNs,
                ax = i.toFloat(), // encode index in ax for tracking
                ay = 0f,
                az = 0f,
                gx = 0f,
                gy = 0f,
                gz = 0f
            )
            buffer.addSample(sample)
            timeNs += 100_000_000L
        }

        // With 45 samples:
        // Window 0 emitted at sample index 19 (size 20: 0..19)
        // Window 1 emitted at sample index 29 (size 20: 10..29)
        // Window 2 emitted at sample index 39 (size 20: 20..39)
        // Window 3 requires 49 (not reached yet)
        assertEquals("Should emit exactly 3 windows for 45 samples with stride 10", 3, emittedWindows.size)

        // Verify shape of each emitted window is strictly [20][6]
        for (w in emittedWindows) {
            assertEquals(20, w.size)
            for (sample in w) {
                assertEquals(6, sample.size)
            }
        }
    }

    @Test
    fun testTimestampJitterResampling() {
        val buffer = IMUBuffer(windowSize = 20, stride = 10, targetDtNs = 100_000_000L)
        var windowEmitted = false

        buffer.onWindowReadyListener = { window, _ ->
            windowEmitted = true
            assertEquals(20, window.size)
            assertEquals(6, window[0].size)
        }

        // Feed jittery timestamps: intervals vary around 100ms (simulating Android sensor jitter at ~10 Hz)
        var timeNs = 1_000_000_000L
        val jitterPattern = listOf(70_000_000L, 120_000_000L, 90_000_000L, 130_000_000L, 80_000_000L, 110_000_000L)

        // Total duration ~ 2.6 seconds (> 20 samples on 100ms grid = 2.0s)
        for (i in 0 until 26) {
            val dt = jitterPattern[i % jitterPattern.size]
            timeNs += dt
            buffer.addSample(
                ImuSample(
                    timestampNs = timeNs,
                    ax = 1.0f,
                    ay = 2.0f,
                    az = 3.0f,
                    gx = 0.1f,
                    gy = 0.2f,
                    gz = 0.3f
                )
            )
        }

        assertTrue("Window should be emitted despite timestamp jitter", windowEmitted)
    }
}
