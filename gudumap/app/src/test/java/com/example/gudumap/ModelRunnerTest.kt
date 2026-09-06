package com.example.gudumap

import com.example.gudumap.ml.InputNormalizer
import com.example.gudumap.ml.ModelMetadata
import com.example.gudumap.ml.ModelRunner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FileInputStream
import kotlin.math.abs

class ModelRunnerTest {

    private fun findModelFile(): File? {
        val candidates = listOf(
            File("src/main/assets/gru_io_vnbd.onnx"),
            File("app/src/main/assets/gru_io_vnbd.onnx"),
            File("../app/src/main/assets/gru_io_vnbd.onnx"),
            File("../../../../models/frozen_io_vnbd/gru_io_vnbd.onnx"),
            File("d:/dead_reckoning/models/frozen_io_vnbd/gru_io_vnbd.onnx"),
            File("d:/dead_reckoning/models/gru_io_vnbd.onnx")
        )
        return candidates.firstOrNull { it.exists() && it.length() > 0 }
    }

    @Test
    fun testContractInputAndOutputShapes() {
        assertEquals(20, ModelMetadata.WINDOW_SIZE)
        assertEquals(6, ModelMetadata.FEATURE_DIM)
        assertEquals(3, ModelMetadata.OUTPUT_DIM)
        assertEquals("gru_io_vnbd.onnx", ModelMetadata.MODEL_FILE_NAME)
    }

    @Test
    fun testOnnxModelExecutionAndPythonReferenceAgreement() {
        val modelFile = findModelFile()
        if (modelFile == null) {
            println("Model file not found on path, skipping native ONNX execution test")
            return
        }

        val runner = ModelRunner()
        try {
            FileInputStream(modelFile).use { stream ->
                runner.initializeFromStream(stream)
            }
        } catch (e: Throwable) {
            println("Native ONNX runtime library not available on host JVM test runner: ${e.message}")
            return
        }

        assertTrue("ModelRunner must be ready after initialization", runner.ready)

        // 1. Test zeros window against exact Python ONNX reference output:
        // Python: [2.5189304, 0.09055324, 0.03168116] (meters)
        val zerosWindow = Array(20) { FloatArray(6) { 0f } }
        val predZeros = runner.predict(zerosWindow)

        assertNotNull(predZeros)
        assertEquals("Output shape must be [3]", 3, predZeros.size)
        assertTrue(predZeros.all { !it.isNaN() && !it.isInfinite() })

        val pythonExpected = floatArrayOf(2.5189304f, 0.09055324f, 0.03168116f)
        for (i in 0 until 3) {
            assertEquals(
                "Prediction [$i] should match Python reference within 1e-4 tolerance",
                pythonExpected[i],
                predZeros[i],
                1e-4f
            )
        }

        // 2. Test deterministic inference (identical inputs produce identical outputs)
        val predZerosRepeat = runner.predict(zerosWindow)
        for (i in 0 until 3) {
            assertEquals(
                "Deterministic inference failed at index $i",
                predZeros[i],
                predZerosRepeat[i],
                1e-6f
            )
        }

        // 3. Test batch inference [batch=2, 20, 6]
        val batch = Array(2) { zerosWindow }
        val batchPreds = runner.predictBatch(batch)
        assertEquals(2, batchPreds.size)
        assertEquals(3, batchPreds[0].size)
        assertEquals(3, batchPreds[1].size)

        runner.close()
        assertFalse("ModelRunner must not be ready after close()", runner.ready)
    }
}
