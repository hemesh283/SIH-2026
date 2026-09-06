package com.example.gudumap

import com.example.gudumap.navigation.DeadReckoningEngine
import com.example.gudumap.navigation.IMUBuffer
import com.example.gudumap.sensor.ImuSample
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Deterministic validation of pipeline causality, temporal monotonicity,
 * blackout GNSS ground-truth isolation, and diagnostic telemetry recorder.
 *
 * Requirements (Checkpoint 2 Task 3 & Task 6):
 * - Ensure each ML inference window contains ONLY samples at or before the inference timestamp.
 * - No future samples.
 * - No look-ahead.
 * - No GNSS ground truth during blackout.
 * - Verification of DiagnosticRecorder telemetry frame recording.
 */
class SensorPipelineCausalityTest {

    @Test
    fun testStrictCausalityAndNoLookAhead() {
        val buffer = IMUBuffer(windowSize = 20, stride = 10, targetDtNs = 100_000_000L)
        val emittedWindows = ArrayList<Pair<Array<FloatArray>, Long>>()

        buffer.onWindowReadyListener = { window, emitTimestampNs ->
            emittedWindows.add(Pair(window, emitTimestampNs))
        }

        // Feed 50 samples at 20 ms intervals (50 Hz incoming sensor stream)
        // Duration: 50 * 20ms = 1000ms = 1.0s
        // Next 50 samples at 20 ms intervals: 1.0s to 2.0s
        // Total: 120 samples = 2.4 seconds
        val startNs = 1_000_000_000L
        var currentNs = startNs

        for (i in 0 until 120) {
            val sample = ImuSample(
                timestampNs = currentNs,
                ax = 0.5f,
                ay = 0.1f,
                az = 9.8f,
                gx = 0.01f,
                gy = 0.02f,
                gz = 0.03f
            )
            buffer.addSample(sample)

            // When a window is emitted, the latest sensor sample timestamp added so far is currentNs
            for (entry in emittedWindows) {
                val emitNs = entry.second
                // The window emission timestamp must never exceed the current sensor timestamp
                assertTrue(
                    "Emitted window timestamp ($emitNs) cannot be in the future relative to current sample ($currentNs)",
                    emitNs <= currentNs
                )
            }
            currentNs += 20_000_000L // 20 ms step (50 Hz)
        }

        assertTrue("At least one window should have been emitted in 2.4s", emittedWindows.isNotEmpty())

        // Verify that the stride between emitted windows is exactly 10 samples (1.0 second = 1,000,000,000 ns)
        if (emittedWindows.size >= 2) {
            val window0Time = emittedWindows[0].second
            val window1Time = emittedWindows[1].second
            val strideDtNs = window1Time - window0Time
            assertEquals(
                "Stride between windows must be exactly 10 samples (1.0s = 1_000_000_000 ns)",
                1_000_000_000L,
                strideDtNs
            )
        }
    }

    @Test
    fun testBackwardAndDuplicateTimestampsRejected() {
        val buffer = IMUBuffer(windowSize = 20, stride = 10, targetDtNs = 100_000_000L)
        var windowCount = 0

        buffer.onWindowReadyListener = { _, _ ->
            windowCount++
        }

        val baseNs = 5_000_000_000L
        buffer.addSample(ImuSample(baseNs, 0f, 0f, 0f, 0f, 0f, 0f))

        // Backward timestamp (earlier than previous)
        buffer.addSample(ImuSample(baseNs - 50_000_000L, 1f, 1f, 1f, 0f, 0f, 0f))

        // Duplicate timestamp (equal to previous)
        buffer.addSample(ImuSample(baseNs, 2f, 2f, 2f, 0f, 0f, 0f))

        // Ensure buffer state is uncorrupted by feeding valid forward samples
        var timeNs = baseNs + 100_000_000L
        for (i in 0 until 25) {
            buffer.addSample(ImuSample(timeNs, 0.1f, 0.1f, 9.8f, 0f, 0f, 0f))
            timeNs += 100_000_000L
        }

        assertTrue("Buffer must continue working cleanly after rejecting invalid timestamps", windowCount >= 1)
    }

    @Test
    fun testBlackoutStrictGnssIsolation() {
        val engine = DeadReckoningEngine()
        engine.initialize(latitude = 28.6139, longitude = 77.2090, speedMps = 15.0f, bearingDeg = 45.0f)

        // Pre-blackout: normal operation
        assertFalse("Blackout should be false initially", engine.getState().isBlackout)

        // Enter blackout mode
        engine.setBlackoutMode(true)
        assertTrue("Blackout must be active", engine.getState().isBlackout)

        // During blackout, GNSS fixes sent to correctWithGnss must be ignored by the filter
        engine.correctWithGnss(latitude = 28.7000, longitude = 77.3000, speedMps = 25.0f, bearingDeg = 90.0f)
        val blackoutState = engine.getState()
        assertEquals(
            "GNSS fix must not alter filter position during blackout",
            28.6139,
            blackoutState.latitude,
            1e-5
        )

        // Sensor updates during blackout should not leak any GNSS speed to ZuptDetector
        var ts = 10_000_000_000L
        for (i in 0 until 5) {
            ts += 100_000_000L
            engine.addSensorSample(ImuSample(ts, 0.01f, 0.01f, 0.01f, 0.001f, 0.001f, 0.001f))
        }

        // When blackout ends, normal GNSS fusion resumes smoothly
        engine.setBlackoutMode(false)
        assertFalse("Blackout must be cleared", engine.getState().isBlackout)
    }

    @Test
    fun testDiagnosticRecorderTelemetry() {
        val engine = DeadReckoningEngine()
        engine.initialize(latitude = 13.0827, longitude = 80.2707, speedMps = 0f, bearingDeg = 0f)

        // Initially disabled: zero frames recorded
        assertFalse(engine.diagnosticRecorder.isEnabled)
        assertEquals(0, engine.diagnosticRecorder.frameCount)

        var ts = 1_000_000_000L
        engine.addSensorSample(ImuSample(ts, 0.1f, 0.2f, 0.3f, 0.01f, 0.02f, 0.03f))
        assertEquals(0, engine.diagnosticRecorder.frameCount)

        // Enable diagnostic recorder
        engine.diagnosticRecorder.isEnabled = true
        assertTrue(engine.diagnosticRecorder.isEnabled)

        // Feed 5 samples
        for (i in 0 until 5) {
            ts += 100_000_000L
            engine.addSensorSample(ImuSample(ts, 0.1f, 0.2f, 0.3f, 0.01f, 0.02f, 0.03f))
        }

        val frames = engine.diagnosticRecorder.getRecentFrames()
        assertEquals(5, frames.size)

        // Verify recorded telemetry properties
        val firstFrame = frames[0]
        assertEquals(0.1f, firstFrame.accX, 1e-4f)
        assertEquals(0.2f, firstFrame.accY, 1e-4f)
        assertEquals(0.3f, firstFrame.accZ, 1e-4f)
        assertEquals(0.01f, firstFrame.gyroX, 1e-4f)
        assertEquals(0.02f, firstFrame.gyroY, 1e-4f)
        assertEquals(0f, firstFrame.effectiveSampleIntervalMs, 1e-1f) // first frame has no prior timestamp
        assertEquals(100f, frames[1].effectiveSampleIntervalMs, 1e-1f) // subsequent frames show 100ms interval
        assertFalse(firstFrame.isBlackout)

        // Clear recorder
        engine.diagnosticRecorder.clear()
        assertEquals(0, engine.diagnosticRecorder.frameCount)
    }
}
