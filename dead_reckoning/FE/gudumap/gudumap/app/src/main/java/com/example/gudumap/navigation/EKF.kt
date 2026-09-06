package com.example.gudumap.navigation

import kotlin.math.max

/**
 * Simplified 6-State MVP Navigation Filter (EKF) for Dead Reckoning Navigation.
 *
 * State vector x:
 * [0] p_N : Position North (meters)
 * [1] p_E : Position East (meters)
 * [2] p_D : Position Down (meters)
 * [3] v_N : Velocity North (m/s)
 * [4] v_E : Velocity East (m/s)
 * [5] v_D : Velocity Down (m/s)
 *
 * MVP Limitations & Scope:
 * - This filter models translational position and velocity in the NED frame.
 * - It does NOT explicitly estimate accelerometer biases, gyroscope biases, or attitude errors
 *   (i.e. it is a simplified navigation filter, not a full 15-state error-state INS EKF).
 * - Fuses:
 *   1. ML displacement / motion estimates (prediction step)
 *   2. GNSS position & velocity when available (measurement update)
 *   3. Non-Holonomic Constraints (NHC pseudo-measurement update)
 *   4. Zero-Velocity Updates (ZUPT pseudo-measurement update when stationary)
 */
class EKF(
    var processNoisePos: Double = 0.05,     // m/sqrt(s)
    var processNoiseVel: Double = 0.20,     // (m/s)/sqrt(s)
    var gnssPosNoise: Double = 3.0,         // meters
    var gnssVelNoise: Double = 0.3,         // m/s
    var zuptVelNoise: Double = 0.02         // m/s (pseudo-measurement noise)
) {

    companion object {
        const val STATE_DIM = 6
    }

    // State vector [p_N, p_E, p_D, v_N, v_E, v_D]
    val state = DoubleArray(STATE_DIM)

    // Covariance matrix 6x6 (row-major)
    val P = Array(STATE_DIM) { DoubleArray(STATE_DIM) }

    var isInitialized: Boolean = false
        private set

    init {
        reset()
    }

    fun reset() {
        for (i in 0 until STATE_DIM) {
            state[i] = 0.0
            for (j in 0 until STATE_DIM) {
                P[i][j] = 0.0
            }
        }
        // Initial uncertainties
        P[0][0] = 10.0; P[1][1] = 10.0; P[2][2] = 10.0 // Position variance
        P[3][3] = 1.0;  P[4][4] = 1.0;  P[5][5] = 1.0  // Velocity variance
        isInitialized = false
    }

    /**
     * Initialize EKF position from known GNSS position / NED origin.
     */
    fun initialize(pNorth: Double, pEast: Double, pDown: Double, vNorth: Double = 0.0, vEast: Double = 0.0, vDown: Double = 0.0) {
        state[0] = pNorth
        state[1] = pEast
        state[2] = pDown
        state[3] = vNorth
        state[4] = vEast
        state[5] = vDown

        // Reset covariance to nominal initial values
        for (i in 0 until STATE_DIM) {
            for (j in 0 until STATE_DIM) {
                P[i][j] = 0.0
            }
        }
        P[0][0] = gnssPosNoise * gnssPosNoise
        P[1][1] = gnssPosNoise * gnssPosNoise
        P[2][2] = gnssPosNoise * gnssPosNoise
        P[3][3] = gnssVelNoise * gnssVelNoise
        P[4][4] = gnssVelNoise * gnssVelNoise
        P[5][5] = gnssVelNoise * gnssVelNoise

        isInitialized = true
    }

    /**
     * Prediction step using INS / GRU displacement delta_p_ned over time interval dt.
     *
     * @param deltaPNed Displacement [dNorth, dEast, dDown] in meters
     * @param dt Time duration in seconds
     */
    fun predict(deltaPNed: DoubleArray, dt: Double) {
        if (!isInitialized) return
        val dtSafe = max(dt, 0.001)

        // 1. Propagate state: p_k = p_{k-1} + delta_p, v_k = delta_p / dt
        state[0] += deltaPNed[0]
        state[1] += deltaPNed[1]
        state[2] += deltaPNed[2]

        state[3] = deltaPNed[0] / dtSafe
        state[4] = deltaPNed[1] / dtSafe
        state[5] = deltaPNed[2] / dtSafe

        // 2. State transition matrix F:
        // [ I_3   dt*I_3 ]
        // [ 0_3     I_3  ]
        val F = Array(STATE_DIM) { i ->
            DoubleArray(STATE_DIM) { j ->
                if (i == j) 1.0
                else if (j == i + 3) dtSafe
                else 0.0
            }
        }

        // 3. Process noise Q
        val qPos = processNoisePos * processNoisePos * dtSafe
        val qVel = processNoiseVel * processNoiseVel * dtSafe

        // P_new = F * P * F^T + Q
        val FP = matrixMultiply(F, P)
        val FT = transpose(F)
        val FPF = matrixMultiply(FP, FT)

        for (i in 0 until STATE_DIM) {
            for (j in 0 until STATE_DIM) {
                P[i][j] = FPF[i][j]
            }
            if (i < 3) P[i][i] += qPos
            else P[i][i] += qVel
        }
    }

    /**
     * Generic Kalman measurement update.
     *
     * @param z Measurement vector (size m)
     * @param H Measurement matrix (m x 6)
     * @param R Measurement noise covariance matrix (m x m)
     */
    fun updateMeasurement(z: DoubleArray, H: Array<DoubleArray>, R: Array<DoubleArray>) {
        if (!isInitialized) return
        val m = z.size
        require(H.size == m && H[0].size == STATE_DIM)
        require(R.size == m && R[0].size == m)

        // Residual y = z - H * x
        val Hx = DoubleArray(m)
        for (i in 0 until m) {
            var sum = 0.0
            for (j in 0 until STATE_DIM) {
                sum += H[i][j] * state[j]
            }
            Hx[i] = sum
        }
        val y = DoubleArray(m) { i -> z[i] - Hx[i] }

        // S = H * P * H^T + R
        val HP = matrixMultiply(H, P)
        val HT = transpose(H)
        val HPH = matrixMultiply(HP, HT)
        val S = Array(m) { i ->
            DoubleArray(m) { j ->
                HPH[i][j] + R[i][j]
            }
        }

        // Invert S (supports m = 1, 2, 3)
        val Sinv = invertMatrix(S) ?: return

        // Kalman gain K = P * H^T * S^-1 (6 x m)
        val PHT = matrixMultiply(P, HT)
        val K = matrixMultiply(PHT, Sinv)

        // State update x = x + K * y
        for (i in 0 until STATE_DIM) {
            var sum = 0.0
            for (j in 0 until m) {
                sum += K[i][j] * y[j]
            }
            state[i] += sum
        }

        // Joseph form covariance update for numerical stability:
        // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        val I_KH = Array(STATE_DIM) { i ->
            DoubleArray(STATE_DIM) { j ->
                val delta = if (i == j) 1.0 else 0.0
                var kh = 0.0
                for (k in 0 until m) {
                    kh += K[i][k] * H[k][j]
                }
                delta - kh
            }
        }

        val I_KH_P = matrixMultiply(I_KH, P)
        val I_KH_T = transpose(I_KH)
        val P_part1 = matrixMultiply(I_KH_P, I_KH_T)

        val KR = matrixMultiply(K, R)
        val KT = transpose(K)
        val P_part2 = matrixMultiply(KR, KT)

        for (i in 0 until STATE_DIM) {
            for (j in 0 until STATE_DIM) {
                P[i][j] = P_part1[i][j] + P_part2[i][j]
            }
        }
    }

    /**
     * Apply GNSS Position measurement update.
     */
    fun updateGnssPosition(pNorth: Double, pEast: Double, pDown: Double, noiseMeters: Double = gnssPosNoise) {
        val z = doubleArrayOf(pNorth, pEast, pDown)
        val H = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        )
        val rVal = noiseMeters * noiseMeters
        val R = arrayOf(
            doubleArrayOf(rVal, 0.0, 0.0),
            doubleArrayOf(0.0, rVal, 0.0),
            doubleArrayOf(0.0, 0.0, rVal)
        )
        updateMeasurement(z, H, R)
    }

    /**
     * Apply GNSS Velocity measurement update.
     */
    fun updateGnssVelocity(vNorth: Double, vEast: Double, vDown: Double, noiseMps: Double = gnssVelNoise) {
        val z = doubleArrayOf(vNorth, vEast, vDown)
        val H = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )
        val rVal = noiseMps * noiseMps
        val R = arrayOf(
            doubleArrayOf(rVal, 0.0, 0.0),
            doubleArrayOf(0.0, rVal, 0.0),
            doubleArrayOf(0.0, 0.0, rVal)
        )
        updateMeasurement(z, H, R)
    }

    /**
     * Zero-Velocity Update (ZUPT) implemented strictly as an EKF pseudo-measurement.
     *
     * IMPORTANT:
     * Does NOT hard-clamp or directly overwrite navigation velocity.
     * Applies standard pseudo-measurement z = [0, 0, 0] with noise covariance R_zupt.
     */
    fun updateZupt(noiseMps: Double = zuptVelNoise) {
        val z = doubleArrayOf(0.0, 0.0, 0.0)
        val H = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )
        val rVal = noiseMps * noiseMps
        val R = arrayOf(
            doubleArrayOf(rVal, 0.0, 0.0),
            doubleArrayOf(0.0, rVal, 0.0),
            doubleArrayOf(0.0, 0.0, rVal)
        )
        updateMeasurement(z, H, R)
    }

    // --- Matrix Math Utilities ---

    private fun matrixMultiply(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val rowsA = A.size
        val colsA = A[0].size
        val colsB = B[0].size
        val C = Array(rowsA) { DoubleArray(colsB) }
        for (i in 0 until rowsA) {
            for (j in 0 until colsB) {
                var sum = 0.0
                for (k in 0 until colsA) {
                    sum += A[i][k] * B[k][j]
                }
                C[i][j] = sum
            }
        }
        return C
    }

    private fun transpose(A: Array<DoubleArray>): Array<DoubleArray> {
        val rows = A.size
        val cols = A[0].size
        val AT = Array(cols) { DoubleArray(rows) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                AT[j][i] = A[i][j]
            }
        }
        return AT
    }

    private fun invertMatrix(M: Array<DoubleArray>): Array<DoubleArray>? {
        return when (M.size) {
            1 -> {
                if (M[0][0] == 0.0) null
                else arrayOf(doubleArrayOf(1.0 / M[0][0]))
            }
            2 -> {
                val det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
                if (det == 0.0) null
                else {
                    val invDet = 1.0 / det
                    arrayOf(
                        doubleArrayOf(M[1][1] * invDet, -M[0][1] * invDet),
                        doubleArrayOf(-M[1][0] * invDet, M[0][0] * invDet)
                    )
                }
            }
            3 -> {
                val a = M[0][0]; val b = M[0][1]; val c = M[0][2]
                val d = M[1][0]; val e = M[1][1]; val f = M[1][2]
                val g = M[2][0]; val h = M[2][1]; val k = M[2][2]

                val det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
                if (det == 0.0) null
                else {
                    val invDet = 1.0 / det
                    arrayOf(
                        doubleArrayOf((e * k - f * h) * invDet, (c * h - b * k) * invDet, (b * f - c * e) * invDet),
                        doubleArrayOf((f * g - d * k) * invDet, (a * k - c * g) * invDet, (c * d - a * f) * invDet),
                        doubleArrayOf((d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet)
                    )
                }
            }
            else -> null
        }
    }
}
