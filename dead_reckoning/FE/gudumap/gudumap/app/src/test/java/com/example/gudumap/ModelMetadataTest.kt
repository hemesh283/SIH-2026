package com.example.gudumap

import com.example.gudumap.ml.ModelMetadata
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream

class ModelMetadataTest {

    @Test
    fun testContractConstants() {
        assertEquals(20, ModelMetadata.WINDOW_SIZE)
        assertEquals(10, ModelMetadata.STRIDE)
        assertEquals(10, ModelMetadata.SAMPLING_RATE_HZ)
        assertEquals(2.0f, ModelMetadata.WINDOW_DURATION_SEC, 1e-5f)
        assertEquals(1.0f, ModelMetadata.STRIDE_DURATION_SEC, 1e-5f)
        assertEquals(6, ModelMetadata.FEATURE_DIM)
        assertEquals(3, ModelMetadata.OUTPUT_DIM)
        assertEquals(9.80665f, ModelMetadata.GRAVITY_MPS2, 1e-5f)
        assertEquals("gru_io_vnbd.onnx", ModelMetadata.MODEL_FILE_NAME)
        assertEquals("io_vnbd_normalization.json", ModelMetadata.NORMALIZATION_FILE_NAME)
    }

    @Test
    fun testFeatureOrdering() {
        val expected = listOf("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
        assertEquals(expected, ModelMetadata.FEATURE_NAMES)
        assertEquals(listOf("dx", "dy", "dz"), ModelMetadata.OUTPUT_NAMES)
    }

    @Test
    fun testDefaultNormalizationValues() {
        assertEquals(6, ModelMetadata.DEFAULT_FEATURE_MEAN.size)
        assertEquals(6, ModelMetadata.DEFAULT_FEATURE_STD.size)
        assertEquals(3, ModelMetadata.DEFAULT_TARGET_MEAN.size)
        assertEquals(3, ModelMetadata.DEFAULT_TARGET_STD.size)

        // Verify exact IO-VNBD frozen values
        assertArrayEquals(
            floatArrayOf(0.0027463287f, -0.012979638f, 0.002593422f, 0.0010569283f, -0.0013110938f, -0.0013259545f),
            ModelMetadata.DEFAULT_FEATURE_MEAN,
            1e-6f
        )
        assertArrayEquals(
            floatArrayOf(0.21921991f, 0.20646203f, 0.09309556f, 0.30223659f, 0.18882969f, 0.11095238f),
            ModelMetadata.DEFAULT_FEATURE_STD,
            1e-6f
        )
        assertArrayEquals(
            floatArrayOf(26.469011f, 0.019715862f, 0.0f),
            ModelMetadata.DEFAULT_TARGET_MEAN,
            1e-6f
        )
        assertArrayEquals(
            floatArrayOf(16.954994f, 1.4200412f, 1.0f),
            ModelMetadata.DEFAULT_TARGET_STD,
            1e-6f
        )

        // Verify std values are strictly positive
        for (std in ModelMetadata.DEFAULT_FEATURE_STD) {
            assertTrue("Feature std must be positive", std > 0f)
        }
        for (std in ModelMetadata.DEFAULT_TARGET_STD) {
            assertTrue("Target std must be positive", std > 0f)
        }
    }

    @Test
    fun testLoadNormalizationFromStream() {
        val json = """
            {
              "feature_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
              "mean": [0.01, 0.02, -0.01, 0.003, 0.04, 0.09],
              "std": [0.07, 0.09, 0.10, 0.31, 0.36, 0.58],
              "target_names": ["dx", "dy", "dz"],
              "target_mean": [0.60, -1.03, -0.47],
              "target_std": [0.75, 0.44, 0.24]
            }
        """.trimIndent()

        val stream = ByteArrayInputStream(json.toByteArray(Charsets.UTF_8))
        val params = ModelMetadata.loadNormalizationFromStream(stream)

        assertEquals(6, params.featureMean.size)
        assertEquals(6, params.featureStd.size)
        assertEquals(3, params.targetMean.size)
        assertEquals(3, params.targetStd.size)

        assertEquals(0.01f, params.featureMean[0], 1e-4f)
        assertEquals(0.75f, params.targetStd[0], 1e-4f)
    }
}
