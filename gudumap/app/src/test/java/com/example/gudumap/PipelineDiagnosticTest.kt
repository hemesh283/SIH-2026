package com.example.gudumap

import com.example.gudumap.navigation.DeadReckoningEngine
import com.example.gudumap.navigation.NavMotionState
import com.example.gudumap.sensor.ImuSample
import com.example.gudumap.sensor.OrientationSample
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FileInputStream
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Diagnostic test harness to pinpoint the exact component causing false distance
 * accumulation during:
 * Stage A: Phone stationary on table
 * Stage B: Pick phone up (vertical movement, zero horizontal travel)
 * Stage C: Rotate/tilt phone in place (angular motion, zero linear travel)
 * Stage D: Hold phone still in hand (hand tremor)
 * Stage E: Actually walk/move (forward travel)
 */
class PipelineDiagnosticTest {

    private fun findModelFile(): File {
        val candidates = listOf(
            File("app/src/main/assets/gru_io_vnbd.onnx"),
            File("src/main/assets/gru_io_vnbd.onnx")
        )
        return candidates.firstOrNull { it.exists() && it.length() > 0 }
            ?: throw IllegalStateException("gru_io_vnbd.onnx not found")
    }

    private fun findNormFile(): File {
        val candidates = listOf(
            File("app/src/main/assets/io_vnbd_normalization.json"),
            File("src/main/assets/io_vnbd_normalization.json")
        )
        return candidates.firstOrNull { it.exists() && it.length() > 0 }
            ?: throw IllegalStateException("io_vnbd_normalization.json not found")
    }

    @Test
    fun runFullPipelineDiagnostic() {
        val engine = DeadReckoningEngine()
        val modelFile = findModelFile()
        FileInputStream(modelFile).use { engine.modelRunner.initializeFromStream(it) }

        engine.initialize(latitude = 11.0168, longitude = 76.9558)

        println("=========================================================================================================")
        println("STARTING GUDUMAP PIPELINE DIAGNOSTIC LOGGING")
        println("=========================================================================================================")
        println(String.format("%-10s | %-12s | %-12s | %-8s | %-18s | %-14s | %-12s | %-12s",
            "STAGE", "ACC_MAG(m/s2)", "GYRO_MAG(r/s)", "ZUPT_ST", "ML_DX/DY/DZ(m)", "ML_MAG(m)", "EKF_SPD(m/s)", "CUMUL_DIST(m)"))
        println("---------------------------------------------------------------------------------------------------------")

        var timeNs = 1_000_000_000L
        val dtNs = 100_000_000L // 10 Hz (100 ms)

        // Helper to run steps and log telemetry
        fun runStage(
            stageName: String,
            durationSec: Float,
            accelGenerator: (Float) -> FloatArray,
            gyroGenerator: (Float) -> FloatArray,
            headingGenerator: (Float) -> Float
        ) {
            val steps = (durationSec * 10).toInt()
            var prevDistance = engine.distanceTravelled

            for (i in 0 until steps) {
                val t = i * 0.1f
                val acc = accelGenerator(t)
                val gyro = gyroGenerator(t)
                val heading = headingGenerator(t)

                val accMag = sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2])
                val gyroMag = sqrt(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2])

                // Orientation
                val headingRad = Math.toRadians(heading.toDouble()).toFloat()
                val rotMatrix = floatArrayOf(
                    cos(headingRad), -sin(headingRad), 0f,
                    sin(headingRad), cos(headingRad), 0f,
                    0f, 0f, 1f
                )
                engine.updateOrientation(
                    OrientationSample(
                        timestampNs = timeNs,
                        rotationMatrix = rotMatrix,
                        quaternion = floatArrayOf(0f, 0f, 0f, 1f),
                        azimuthRad = headingRad,
                        pitchRad = 0f,
                        rollRad = 0f
                    )
                )

                // IMU sample
                engine.addSensorSample(
                    ImuSample(
                        timestampNs = timeNs,
                        ax = acc[0], ay = acc[1], az = acc[2],
                        gx = gyro[0], gy = gyro[1], gz = gyro[2]
                    )
                )

                val state = engine.getState()
                val distIncr = state.distanceTravelled - prevDistance
                prevDistance = state.distanceTravelled

                // Check raw ML prediction on latest window directly if available
                val rawMl = if (engine.imuBuffer.size >= 20) {
                    // Extract raw prediction from modelRunner to see what GRU outputs on this window
                    val dummyWindow = Array(20) {
                        floatArrayOf(
                            acc[0] / 9.80665f, acc[1] / 9.80665f, acc[2] / 9.80665f,
                            gyro[0], gyro[1], gyro[2]
                        )
                    }
                    try {
                        engine.modelRunner.predict(dummyWindow)
                    } catch (_: Exception) {
                        floatArrayOf(0f, 0f, 0f)
                    }
                } else {
                    floatArrayOf(0f, 0f, 0f)
                }
                val mlMag = sqrt(rawMl[0] * rawMl[0] + rawMl[1] * rawMl[1] + rawMl[2] * rawMl[2])

                // Log every 500ms (every 5 samples)
                if (i % 5 == 0 || i == steps - 1) {
                    println(String.format(
                        "%-10s | %12.4f | %12.4f | %-8b | [%.2f, %.2f, %.2f] | %14.4f | %12.4f | %12.4f",
                        stageName,
                        accMag,
                        gyroMag,
                        engine.isStationary,
                        rawMl[0], rawMl[1], rawMl[2],
                        mlMag,
                        state.speed,
                        state.distanceTravelled
                    ))
                }

                timeNs += dtNs
            }
        }

        // =========================================================================
        // STAGE A: Phone stationary on table (3 seconds)
        // =========================================================================
        println("\n>>> STAGE A: Stationary on table (True motion: 0.0m)")
        runStage(
            stageName = "STAGE_A",
            durationSec = 3.0f,
            accelGenerator = { floatArrayOf(0.01f, 0.01f, 0.01f) },
            gyroGenerator = { floatArrayOf(0.002f, 0.002f, 0.002f) },
            headingGenerator = { 0f }
        )
        val distA = engine.distanceTravelled
        println("STAGE A RESULT: Distance = $distA meters (Expected: 0.0m)")

        // =========================================================================
        // STAGE B: Pick phone up (2.5 seconds: vertical lift, zero horizontal motion)
        // =========================================================================
        println("\n>>> STAGE B: Pick phone up (True horizontal motion: 0.0m)")
        runStage(
            stageName = "STAGE_B",
            durationSec = 2.5f,
            accelGenerator = { t ->
                // Lift acceleration pulse on Z (up/down), near-zero X/Y
                val az = if (t < 0.5f) 1.2f else if (t < 1.0f) -1.2f else 0.05f
                floatArrayOf(0.08f, 0.05f, az)
            },
            gyroGenerator = { t ->
                // Slight angular wobble during pickup
                val gy = if (t < 1.0f) 0.35f else 0.02f
                floatArrayOf(0.05f, gy, 0.02f)
            },
            headingGenerator = { 0f }
        )
        val distB = engine.distanceTravelled - distA
        println("STAGE B RESULT: Delta Distance = $distB meters (Expected: 0.0m)")

        // =========================================================================
        // STAGE C: Rotate/tilt phone in place (3 seconds: angular rotation, zero travel)
        // =========================================================================
        println("\n>>> STAGE C: Rotate/tilt in place (True horizontal motion: 0.0m)")
        runStage(
            stageName = "STAGE_C",
            durationSec = 3.0f,
            accelGenerator = { floatArrayOf(0.12f, 0.10f, 0.15f) },
            gyroGenerator = { t ->
                // Active rotation in yaw (e.g. looking around 360 deg or turning phone)
                floatArrayOf(0.10f, 0.15f, 1.20f)
            },
            headingGenerator = { t -> t * 60f } // turning
        )
        val distC = engine.distanceTravelled - distA - distB
        println("STAGE C RESULT: Delta Distance = $distC meters (Expected: 0.0m)")

        // =========================================================================
        // STAGE D: Hold phone still in hand (3 seconds: hand tremor)
        // =========================================================================
        println("\n>>> STAGE D: Hold still in hand (True motion: 0.0m)")
        runStage(
            stageName = "STAGE_D",
            durationSec = 3.0f,
            accelGenerator = { t ->
                // Subtle hand tremor
                floatArrayOf(0.08f * sin(t * 10f), 0.06f * cos(t * 10f), 0.05f)
            },
            gyroGenerator = { t ->
                floatArrayOf(0.03f * cos(t * 8f), 0.03f * sin(t * 8f), 0.02f)
            },
            headingGenerator = { 180f }
        )
        val distD = engine.distanceTravelled - distA - distB - distC
        println("STAGE D RESULT: Delta Distance = $distD meters (Expected: 0.0m)")

        // =========================================================================
        // STAGE E: Actually walk / move (5 seconds: walking forward at 1.2 m/s)
        // =========================================================================
        println("\n>>> STAGE E: Real walking forward (True motion: ~6.0m)")
        runStage(
            stageName = "STAGE_E",
            durationSec = 5.0f,
            accelGenerator = { t ->
                // Walking step acceleration pattern (forward + vertical bounce)
                val axForward = 0.5f + 0.8f * sin(t * 12f)
                val azBounce = 0.9f * cos(t * 12f)
                floatArrayOf(axForward, 0.1f, azBounce)
            },
            gyroGenerator = { t ->
                floatArrayOf(0.15f * sin(t * 12f), 0.25f * cos(t * 12f), 0.05f)
            },
            headingGenerator = { 0f } // Walking North
        )
        val distE = engine.distanceTravelled - distA - distB - distC - distD
        println("STAGE E RESULT: Delta Distance = $distE meters (Expected: ~5-6m)")

        println("=========================================================================================================")
        println("DIAGNOSTIC SUMMARY TOTALS:")
        println("Stage A (Table):            Delta = $distA m")
        println("Stage B (Pickup):           Delta = $distB m")
        println("Stage C (Rotate in place):  Delta = $distC m")
        println("Stage D (Hand held still):  Delta = $distD m")
        println("Stage E (Real Walking):     Delta = $distE m")
        println("Total Distance:             ${engine.distanceTravelled} m")
        println("=========================================================================================================")
    }

    @Test
    fun testStageA_TableStationary() {
        val engine = DeadReckoningEngine()
        FileInputStream(findModelFile()).use { engine.modelRunner.initializeFromStream(it) }
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        for (i in 0 until 30) {
            engine.addSensorSample(ImuSample(timeNs, 0.01f, 0.01f, 0.01f, 0.002f, 0.002f, 0.002f))
            timeNs += 100_000_000L
        }

        assertEquals(0.0, engine.distanceTravelled, 0.001)
        assertEquals(0.0f, engine.getState().speed, 0.001f)
        assertTrue(engine.zuptDetector.isStationary)
        assertEquals(NavMotionState.STATIONARY, engine.zuptDetector.motionState)
    }

    @Test
    fun testStageB_PickPhoneVertically_rejectsImplausiblePrediction() {
        val engine = DeadReckoningEngine()
        FileInputStream(findModelFile()).use { engine.modelRunner.initializeFromStream(it) }
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        // 1. Initial 1 second still
        for (i in 0 until 10) {
            engine.addSensorSample(ImuSample(timeNs, 0.01f, 0.01f, 0.01f, 0.002f, 0.002f, 0.002f))
            timeNs += 100_000_000L
        }

        // 2. Vertical pickup (az = 1.2 m/s^2, low horizontal acc, small tilt gyro)
        for (i in 0 until 25) {
            val t = i * 0.1f
            val az = if (t < 0.5f) 1.2f else if (t < 1.0f) -1.2f else 0.05f
            val gy = if (t < 1.0f) 0.35f else 0.02f
            engine.addSensorSample(ImuSample(timeNs, 0.08f, 0.05f, az, 0.05f, gy, 0.02f))
            timeNs += 100_000_000L
        }

        assertEquals("Vertical pickup must accumulate 0 distance", 0.0, engine.distanceTravelled, 0.05)
        assertEquals("EKF velocity must remain 0 after pickup", 0.0f, engine.getState().speed, 0.05f)
    }

    @Test
    fun testStageC_RotatePhoneInPlace_stateRotatingInPlace_orientationUpdates() {
        val engine = DeadReckoningEngine()
        FileInputStream(findModelFile()).use { engine.modelRunner.initializeFromStream(it) }
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        // Pure yaw rotation at 1.2 rad/s in place (e.g. rotating 180 degrees over 3 seconds)
        for (i in 0 until 30) {
            val t = i * 0.1f
            val headingDeg = t * 60f
            val headingRad = Math.toRadians(headingDeg.toDouble()).toFloat()
            engine.updateOrientation(
                OrientationSample(
                    timestampNs = timeNs,
                    rotationMatrix = floatArrayOf(cos(headingRad), -sin(headingRad), 0f, sin(headingRad), cos(headingRad), 0f, 0f, 0f, 1f),
                    quaternion = floatArrayOf(0f, 0f, 0f, 1f),
                    azimuthRad = headingRad,
                    pitchRad = 0f,
                    rollRad = 0f
                )
            )
            // Low horizontal acc (0.12, 0.10) but active yaw gyro (1.20 rad/s)
            engine.addSensorSample(ImuSample(timeNs, 0.12f, 0.10f, 0.15f, 0.10f, 0.15f, 1.20f))
            timeNs += 100_000_000L
        }

        assertEquals("In-place rotation state must be ROTATING_IN_PLACE", NavMotionState.ROTATING_IN_PLACE, engine.zuptDetector.motionState)
        assertEquals("In-place rotation must not accumulate distance", 0.0, engine.distanceTravelled, 0.05)
        assertEquals("EKF velocity must remain 0 during rotation in place", 0.0f, engine.getState().speed, 0.05f)
        assertTrue("Heading must update continuously during rotation", engine.getState().heading > 100f)
    }

    @Test
    fun testStageD_HoldStillAfterPickup_promptZeroVelocityAndNoDistance() {
        val engine = DeadReckoningEngine()
        FileInputStream(findModelFile()).use { engine.modelRunner.initializeFromStream(it) }
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        // 1. Pickup motion
        for (i in 0 until 15) {
            engine.addSensorSample(ImuSample(timeNs, 0.08f, 0.05f, 1.2f, 0.05f, 0.35f, 0.02f))
            timeNs += 100_000_000L
        }
        // 2. Hand held still (tremor: <0.1 m/s^2 acc, <0.04 rad/s gyro)
        for (i in 0 until 30) {
            val t = i * 0.1f
            val ax = 0.08f * sin(t * 10f)
            val ay = 0.06f * cos(t * 10f)
            engine.addSensorSample(ImuSample(timeNs, ax, ay, 0.05f, 0.03f, 0.03f, 0.02f))
            timeNs += 100_000_000L
        }

        assertEquals("Holding still in hand must not accumulate distance", 0.0, engine.distanceTravelled, 0.05)
        assertEquals("Speed must remain 0 while holding still in hand", 0.0f, engine.getState().speed, 0.05f)
        assertTrue("Must transition promptly to stationary", engine.zuptDetector.isStationary)
    }

    @Test
    fun testStageE_GenuineWalking_movementDetectedAndDistanceIncreases() {
        val engine = DeadReckoningEngine()
        FileInputStream(findModelFile()).use { engine.modelRunner.initializeFromStream(it) }
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        // 5 seconds of walking steps (ax forward, az bounce, arm swing gyro)
        for (i in 0 until 50) {
            val t = i * 0.1f
            val axForward = 0.5f + 0.8f * sin(t * 12f)
            val azBounce = 0.9f * cos(t * 12f)
            val gx = 0.15f * sin(t * 12f)
            val gy = 0.25f * cos(t * 12f)
            engine.addSensorSample(ImuSample(timeNs, axForward, 0.1f, azBounce, gx, gy, 0.05f))
            timeNs += 100_000_000L
        }

        assertTrue("Walking must accumulate positive distance", engine.distanceTravelled > 3.0)
        assertTrue("Walking distance must be bounded by physical kinematics (~5-16m, not >100m)", engine.distanceTravelled < 16.0)
        assertTrue("Walking speed must be positive and reasonable", engine.getState().speed > 0.5f)
        assertEquals("Motion state must be MOVING", NavMotionState.MOVING, engine.zuptDetector.motionState)
    }
}
