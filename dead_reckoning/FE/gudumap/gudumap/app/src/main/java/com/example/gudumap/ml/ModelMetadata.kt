package com.example.gudumap.ml

import android.content.Context
import org.json.JSONObject
import java.io.InputStream

/**
 * Contract constants and metadata for GRU Dead Reckoning Model.
 *
 * Model Contract (IO-VNBD Native 10 Hz):
 * - Input shape: [batch, 20, 6]
 * - Feature order: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
 * - Sampling rate: 10 Hz
 * - Window size: 20 samples = 2.0 seconds
 * - Stride: 10 samples = 1.0 second (inference scheduling)
 * - Output shape: [batch, 3] -> [dx, dy, dz] local body frame displacement in meters
 */
object ModelMetadata {

    const val WINDOW_SIZE = 20
    const val STRIDE = 10
    const val SAMPLING_RATE_HZ = 10
    const val WINDOW_DURATION_SEC = 2.0f
    const val STRIDE_DURATION_SEC = 1.0f

    const val FEATURE_DIM = 6
    const val OUTPUT_DIM = 3

    const val MODEL_FILE_NAME = "gru_io_vnbd.onnx"
    const val NORMALIZATION_FILE_NAME = "io_vnbd_normalization.json"
    const val METADATA_FILE_NAME = "model_metadata.json"

    // Standard gravity for m/s^2 to g conversion
    const val GRAVITY_MPS2 = 9.80665f

    val FEATURE_NAMES = listOf(
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z"
    )

    val OUTPUT_NAMES = listOf("dx", "dy", "dz")

    // Default normalization statistics from training (models/frozen_io_vnbd/io_vnbd_normalization.json)
    val DEFAULT_FEATURE_MEAN = floatArrayOf(
        0.0027463287f,
        -0.012979638f,
        0.002593422f,
        0.0010569283f,
        -0.0013110938f,
        -0.0013259545f
    )

    val DEFAULT_FEATURE_STD = floatArrayOf(
        0.21921991f,
        0.20646203f,
        0.09309556f,
        0.30223659f,
        0.18882969f,
        0.11095238f
    )

    val DEFAULT_TARGET_MEAN = floatArrayOf(
        26.469011f,
        0.019715862f,
        0.0f
    )

    val DEFAULT_TARGET_STD = floatArrayOf(
        16.954994f,
        1.4200412f,
        1.0f
    )

    /**
     * Reads normalization values from an input stream (e.g. assets/io_vnbd_normalization.json).
     * Supports both Android OS JSONObject and pure Kotlin fallback for host JVM tests.
     */
    fun loadNormalizationFromStream(inputStream: InputStream): NormalizationParameters {
        val jsonText = inputStream.bufferedReader().use { it.readText() }
        return try {
            val json = JSONObject(jsonText)
            val meanArray = json.getJSONArray("mean")
            val stdArray = json.getJSONArray("std")
            val targetMeanArray = json.getJSONArray("target_mean")
            val targetStdArray = json.getJSONArray("target_std")

            val featureMean = FloatArray(FEATURE_DIM) { i -> meanArray.getDouble(i).toFloat() }
            val featureStd = FloatArray(FEATURE_DIM) { i -> stdArray.getDouble(i).toFloat() }
            val targetMean = FloatArray(OUTPUT_DIM) { i -> targetMeanArray.getDouble(i).toFloat() }
            val targetStd = FloatArray(OUTPUT_DIM) { i -> targetStdArray.getDouble(i).toFloat() }

            NormalizationParameters(featureMean, featureStd, targetMean, targetStd)
        } catch (_: Throwable) {
            // Pure Kotlin fallback for host unit-test environments without Android framework runtime
            val featureMean = extractArray(jsonText, "mean", FEATURE_DIM, DEFAULT_FEATURE_MEAN)
            val featureStd = extractArray(jsonText, "std", FEATURE_DIM, DEFAULT_FEATURE_STD)
            val targetMean = extractArray(jsonText, "target_mean", OUTPUT_DIM, DEFAULT_TARGET_MEAN)
            val targetStd = extractArray(jsonText, "target_std", OUTPUT_DIM, DEFAULT_TARGET_STD)
            NormalizationParameters(featureMean, featureStd, targetMean, targetStd)
        }
    }

    private fun extractArray(json: String, key: String, expectedDim: Int, default: FloatArray): FloatArray {
        val keyIndex = json.indexOf("\"$key\"")
        if (keyIndex == -1) return default
        val startBracket = json.indexOf('[', keyIndex)
        val endBracket = json.indexOf(']', startBracket)
        if (startBracket == -1 || endBracket == -1) return default
        val numbers = json.substring(startBracket + 1, endBracket)
            .split(',')
            .mapNotNull { it.trim().toFloatOrNull() }
        if (numbers.size != expectedDim) return default
        return numbers.toFloatArray()
    }

    /**
     * Load normalization parameters from Context assets, or return default values if unavailable.
     */
    fun loadNormalization(context: Context?): NormalizationParameters {
        if (context == null) {
            return NormalizationParameters(
                DEFAULT_FEATURE_MEAN,
                DEFAULT_FEATURE_STD,
                DEFAULT_TARGET_MEAN,
                DEFAULT_TARGET_STD
            )
        }
        return try {
            context.assets.open(NORMALIZATION_FILE_NAME).use { stream ->
                loadNormalizationFromStream(stream)
            }
        } catch (_: Exception) {
            NormalizationParameters(
                DEFAULT_FEATURE_MEAN,
                DEFAULT_FEATURE_STD,
                DEFAULT_TARGET_MEAN,
                DEFAULT_TARGET_STD
            )
        }
    }
}

data class NormalizationParameters(
    val featureMean: FloatArray,
    val featureStd: FloatArray,
    val targetMean: FloatArray,
    val targetStd: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as NormalizationParameters
        return featureMean.contentEquals(other.featureMean) &&
                featureStd.contentEquals(other.featureStd) &&
                targetMean.contentEquals(other.targetMean) &&
                targetStd.contentEquals(other.targetStd)
    }

    override fun hashCode(): Int {
        var result = featureMean.contentHashCode()
        result = 31 * result + featureStd.contentHashCode()
        result = 31 * result + targetMean.contentHashCode()
        result = 31 * result + targetStd.contentHashCode()
        return result
    }
}
