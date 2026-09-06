package com.example.gudumap

import com.example.gudumap.ml.InputNormalizer
import com.example.gudumap.ml.ModelMetadata
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class InputNormalizerTest {

    private val normalizer = InputNormalizer()

    @Test
    fun testValidWindowPassesValidation() {
        val validWindow = Array(20) { FloatArray(6) { 0f } }
        normalizer.validateWindow(validWindow) // Should not throw
    }

    @Test
    fun testInsufficientWindowRejection() {
        // Less than 20 samples
        val shortWindow = Array(10) { FloatArray(6) }
        try {
            normalizer.validateWindow(shortWindow)
            fail("Expected IllegalArgumentException for window size 10")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Invalid window size"))
        }

        // More than 20 samples
        val longWindow = Array(25) { FloatArray(6) }
        try {
            normalizer.validateWindow(longWindow)
            fail("Expected IllegalArgumentException for window size 25")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Invalid window size"))
        }
    }

    @Test
    fun testWrongFeatureDimensionRejection() {
        // 5 features instead of 6
        val fiveDim = Array(20) { FloatArray(5) }
        try {
            normalizer.validateWindow(fiveDim)
            fail("Expected IllegalArgumentException for feature dimension 5")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Invalid feature count"))
        }

        // 7 features instead of 6
        val sevenDim = Array(20) { FloatArray(7) }
        try {
            normalizer.validateWindow(sevenDim)
            fail("Expected IllegalArgumentException for feature dimension 7")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Invalid feature count"))
        }
    }

    @Test
    fun testNanRejection() {
        val nanWindow = Array(20) { FloatArray(6) }
        nanWindow[5][2] = Float.NaN

        try {
            normalizer.validateWindow(nanWindow)
            fail("Expected IllegalArgumentException for NaN value")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("non-finite"))
        }
    }

    @Test
    fun testPositiveInfinityRejection() {
        val infWindow = Array(20) { FloatArray(6) }
        infWindow[10][0] = Float.POSITIVE_INFINITY

        try {
            normalizer.validateWindow(infWindow)
            fail("Expected IllegalArgumentException for +Inf value")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("non-finite"))
        }
    }

    @Test
    fun testNegativeInfinityRejection() {
        val ninfWindow = Array(20) { FloatArray(6) }
        ninfWindow[19][5] = Float.NEGATIVE_INFINITY

        try {
            normalizer.validateWindow(ninfWindow)
            fail("Expected IllegalArgumentException for -Inf value")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("non-finite"))
        }
    }

    @Test
    fun testUnitConversionLinearAcceleration() {
        val accMps2 = floatArrayOf(9.80665f, -9.80665f, 0f)
        val gyroRads = floatArrayOf(0.1f, -0.2f, 0.3f)

        val sample = normalizer.convertSensorUnits(accMps2, gyroRads)
        assertEquals(6, sample.size)

        // 9.80665 m/s^2 -> 1.0 g
        assertEquals(1.0f, sample[0], 1e-5f)
        // -9.80665 m/s^2 -> -1.0 g
        assertEquals(-1.0f, sample[1], 1e-5f)
        assertEquals(0.0f, sample[2], 1e-5f)

        // Gyroscope in rad/s unchanged
        assertEquals(0.1f, sample[3], 1e-5f)
        assertEquals(-0.2f, sample[4], 1e-5f)
        assertEquals(0.3f, sample[5], 1e-5f)
    }

    @Test
    fun testNormalizationFormula() {
        // If input equals mean, normalized should be 0.0
        val meanInput = Array(20) {
            normalizer.params.featureMean.clone()
        }
        val normWindow = normalizer.normalizeWindow(meanInput)

        for (i in 0 until 20) {
            for (j in 0 until 6) {
                assertEquals("Value equal to mean should normalize to 0", 0f, normWindow[i][j], 1e-5f)
            }
        }
    }

    @Test
    fun testTargetDenormalization() {
        // If normalized target is 0, output should be target_mean
        val zeroNorm = floatArrayOf(0f, 0f, 0f)
        val meters = normalizer.denormalizeTarget(zeroNorm)

        assertEquals(normalizer.params.targetMean[0], meters[0], 1e-5f)
        assertEquals(normalizer.params.targetMean[1], meters[1], 1e-5f)
        assertEquals(normalizer.params.targetMean[2], meters[2], 1e-5f)

        // If normalized target is 1, output should be target_mean + target_std
        val oneNorm = floatArrayOf(1f, 1f, 1f)
        val metersOne = normalizer.denormalizeTarget(oneNorm)
        assertEquals(normalizer.params.targetMean[0] + normalizer.params.targetStd[0], metersOne[0], 1e-5f)
    }
}
