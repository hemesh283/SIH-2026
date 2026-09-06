package com.example.gudumap.ml

/**
 * Handles input validation, unit conversion, feature normalization,
 * and target displacement denormalization according to the ML Deployment Contract.
 */
class InputNormalizer(
    val params: NormalizationParameters = NormalizationParameters(
        ModelMetadata.DEFAULT_FEATURE_MEAN,
        ModelMetadata.DEFAULT_FEATURE_STD,
        ModelMetadata.DEFAULT_TARGET_MEAN,
        ModelMetadata.DEFAULT_TARGET_STD
    )
) {

    companion object {
        fun fromAssets(context: android.content.Context): InputNormalizer {
            return InputNormalizer(ModelMetadata.loadNormalization(context))
        }
    }

    /**
     * Validate an input window.
     * Must have shape [20][6] and all finite values (no NaN or Inf).
     *
     * @param window Array of shape [WINDOW_SIZE][FEATURE_DIM]
     * @throws IllegalArgumentException if shape is incorrect or contains NaN/Inf.
     */
    fun validateWindow(window: Array<FloatArray>) {
        if (window.size != ModelMetadata.WINDOW_SIZE) {
            throw IllegalArgumentException(
                "Invalid window size ${window.size}, expected ${ModelMetadata.WINDOW_SIZE}"
            )
        }
        for (i in window.indices) {
            val sample = window[i]
            if (sample.size != ModelMetadata.FEATURE_DIM) {
                throw IllegalArgumentException(
                    "Invalid feature count ${sample.size} at index $i, expected ${ModelMetadata.FEATURE_DIM}"
                )
            }
            for (j in sample.indices) {
                val value = sample[j]
                if (value.isNaN() || value.isInfinite()) {
                    throw IllegalArgumentException(
                        "Input window contains non-finite value ($value) at [$i, $j]"
                    )
                }
            }
        }
    }

    /**
     * Converts a single IMU sample from Android sensor units:
     * - Linear acceleration: m/s^2 -> g (acc_g = acc_mps2 / 9.80665)
     * - Gyroscope: rad/s -> rad/s (unchanged)
     *
     * @param accMps2 FloatArray of 3 elements [ax, ay, az] in m/s^2
     * @param gyroRads FloatArray of 3 elements [gx, gy, gz] in rad/s
     * @return FloatArray of 6 elements [acc_x_g, acc_y_g, acc_z_g, gyro_x, gyro_y, gyro_z]
     */
    fun convertSensorUnits(accMps2: FloatArray, gyroRads: FloatArray): FloatArray {
        require(accMps2.size == 3) { "accMps2 must have 3 elements" }
        require(gyroRads.size == 3) { "gyroRads must have 3 elements" }

        val sample = FloatArray(ModelMetadata.FEATURE_DIM)
        sample[0] = accMps2[0] / ModelMetadata.GRAVITY_MPS2
        sample[1] = accMps2[1] / ModelMetadata.GRAVITY_MPS2
        sample[2] = accMps2[2] / ModelMetadata.GRAVITY_MPS2
        sample[3] = gyroRads[0]
        sample[4] = gyroRads[1]
        sample[5] = gyroRads[2]
        return sample
    }

    /**
     * Normalize a single 20x6 window using training statistics.
     * Normalized = (input - feature_mean) / feature_std.
     *
     * @param window Raw or unit-converted window of shape [20][6]
     * @return Normalized window of shape [20][6]
     */
    fun normalizeWindow(window: Array<FloatArray>): Array<FloatArray> {
        validateWindow(window)
        val normalized = Array(ModelMetadata.WINDOW_SIZE) { FloatArray(ModelMetadata.FEATURE_DIM) }

        for (i in 0 until ModelMetadata.WINDOW_SIZE) {
            for (j in 0 until ModelMetadata.FEATURE_DIM) {
                val value = window[i][j]
                normalized[i][j] = (value - params.featureMean[j]) / params.featureStd[j]
            }
        }
        return normalized
    }

    /**
     * Flattens normalized window into 1D float array of size [batch * 20 * 6].
     */
    fun flattenBatch(batch: Array<Array<FloatArray>>): FloatArray {
        val batchSize = batch.size
        val flat = FloatArray(batchSize * ModelMetadata.WINDOW_SIZE * ModelMetadata.FEATURE_DIM)
        var idx = 0
        for (b in 0 until batchSize) {
            val window = batch[b]
            for (i in 0 until ModelMetadata.WINDOW_SIZE) {
                for (j in 0 until ModelMetadata.FEATURE_DIM) {
                    flat[idx++] = window[i][j]
                }
            }
        }
        return flat
    }

    /**
     * Denormalizes model raw output back to physical displacement in meters.
     * target_meters = normalized_target * target_std + target_mean.
     *
     * @param normalizedTarget Array of 3 elements [norm_dx, norm_dy, norm_dz]
     * @return Displacement in meters [dx, dy, dz] in initial window local body frame
     */
    fun denormalizeTarget(normalizedTarget: FloatArray): FloatArray {
        require(normalizedTarget.size == ModelMetadata.OUTPUT_DIM) {
            "Expected output dim ${ModelMetadata.OUTPUT_DIM}, got ${normalizedTarget.size}"
        }
        val targetMeters = FloatArray(ModelMetadata.OUTPUT_DIM)
        for (i in 0 until ModelMetadata.OUTPUT_DIM) {
            targetMeters[i] = normalizedTarget[i] * params.targetStd[i] + params.targetMean[i]
        }
        return targetMeters
    }
}
