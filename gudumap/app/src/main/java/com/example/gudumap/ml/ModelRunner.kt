package com.example.gudumap.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.FloatBuffer

/**
 * Executes local on-device neural network inference for Dead Reckoning using ONNX Runtime.
 *
 * Model Contract (IO-VNBD Native 10 Hz):
 * - Model file: gru_io_vnbd.onnx
 * - Input node shape: [batch, 20, 6]
 * - Output node shape: [batch, 3]
 * - Local deployment: All inference runs locally on the device (no cloud backend required).
 */
class ModelRunner(
    private val normalizer: InputNormalizer = InputNormalizer()
) : AutoCloseable {

    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String = "input"
    private var isInitialized: Boolean = false

    /**
     * Initialize ONNX Runtime session from Android asset manager.
     */
    fun initializeFromAssets(
        context: Context,
        assetFileName: String = ModelMetadata.MODEL_FILE_NAME
    ) {
        context.assets.open(assetFileName).use { inputStream ->
            initializeFromStream(inputStream)
        }
    }

    /**
     * Initialize ONNX Runtime session from InputStream (file or asset).
     */
    fun initializeFromStream(inputStream: InputStream) {
        val buffer = ByteArrayOutputStream()
        val data = ByteArray(16384)
        var nRead: Int
        while (inputStream.read(data, 0, data.size).also { nRead = it } != -1) {
            buffer.write(data, 0, nRead)
        }
        buffer.flush()
        initializeFromBytes(buffer.toByteArray())
    }

    /**
     * Initialize ONNX Runtime session from raw model bytes.
     */
    fun initializeFromBytes(modelBytes: ByteArray) {
        close()
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(2)
        val sess = env.createSession(modelBytes, opts)

        this.environment = env
        this.session = sess
        this.inputName = sess.inputNames.iterator().next()
        this.isInitialized = true
    }

    val ready: Boolean
        get() = isInitialized && session != null

    /**
     * Runs forward inference on a single 20x6 window.
     *
     * @param window Sequential IMU samples with shape [20][6]
     *               Features: [acc_x (g), acc_y (g), acc_z (g), gyro_x (rad/s), gyro_y (rad/s), gyro_z (rad/s)]
     * @return Physical displacement [dx, dy, dz] in meters in the initial window body frame
     */
    fun predict(window: Array<FloatArray>): FloatArray {
        check(ready) { "ModelRunner is not initialized. Call initializeFromAssets or initializeFromBytes first." }

        // 1. Validate & Normalize
        val normalizedWindow = normalizer.normalizeWindow(window)

        // 2. Wrap into batch of 1: [1, 20, 6]
        val batch = arrayOf(normalizedWindow)
        val flatData = normalizer.flattenBatch(batch)

        val env = environment ?: throw IllegalStateException("OrtEnvironment is null")
        val sess = session ?: throw IllegalStateException("OrtSession is null")

        val shape = longArrayOf(1, ModelMetadata.WINDOW_SIZE.toLong(), ModelMetadata.FEATURE_DIM.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatData), shape)

        tensor.use { inTensor ->
            val output = sess.run(mapOf(inputName to inTensor))
            output.use { ortOutputs ->
                @Suppress("UNCHECKED_CAST")
                val rawOutput = ortOutputs[0].value as Array<FloatArray>
                val normalizedDisplacement = rawOutput[0]

                // 3. Denormalize to meters
                return normalizer.denormalizeTarget(normalizedDisplacement)
            }
        }
    }

    /**
     * Runs forward batch inference on multiple sequential 20x6 windows.
     *
     * @param windows Array of windows with shape [batch_size, 20, 6]
     * @return Array of displacements in meters [batch_size, 3]
     */
    fun predictBatch(windows: Array<Array<FloatArray>>): Array<FloatArray> {
        check(ready) { "ModelRunner is not initialized." }
        val batchSize = windows.size
        require(batchSize > 0) { "Batch size must be greater than 0" }

        val normalizedBatch = Array(batchSize) { b ->
            normalizer.normalizeWindow(windows[b])
        }
        val flatData = normalizer.flattenBatch(normalizedBatch)

        val env = environment ?: throw IllegalStateException("OrtEnvironment is null")
        val sess = session ?: throw IllegalStateException("OrtSession is null")

        val shape = longArrayOf(batchSize.toLong(), ModelMetadata.WINDOW_SIZE.toLong(), ModelMetadata.FEATURE_DIM.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatData), shape)

        tensor.use { inTensor ->
            val output = sess.run(mapOf(inputName to inTensor))
            output.use { ortOutputs ->
                @Suppress("UNCHECKED_CAST")
                val rawOutputs = ortOutputs[0].value as Array<FloatArray>
                val results = Array(batchSize) { FloatArray(ModelMetadata.OUTPUT_DIM) }
                for (b in 0 until batchSize) {
                    results[b] = normalizer.denormalizeTarget(rawOutputs[b])
                }
                return results
            }
        }
    }

    override fun close() {
        try {
            session?.close()
        } catch (_: Exception) {}
        session = null

        try {
            environment?.close()
        } catch (_: Exception) {}
        environment = null

        isInitialized = false
    }
}
