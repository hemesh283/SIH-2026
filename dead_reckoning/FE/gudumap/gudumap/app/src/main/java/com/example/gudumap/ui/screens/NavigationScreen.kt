package com.example.gudumap.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.location.Location

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll

import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

import androidx.core.content.ContextCompat

import com.example.gudumap.sensors.AccelerometerData
import com.example.gudumap.sensors.DeadReckoningEngine
import com.example.gudumap.sensors.DeadReckoningState
import com.example.gudumap.sensors.GyroscopeData
import com.example.gudumap.sensors.LocationManager
import com.example.gudumap.sensors.MagnetometerData
import com.example.gudumap.sensors.SensorFusionManager
import com.example.gudumap.sensors.SensorManager
import com.example.gudumap.navigation.DeadReckoningEngine as ProductionNavigationEngine
import com.example.gudumap.sensor.SensorManager as HighRateImuSensorManager

import com.example.gudumap.ui.components.MapView

import java.util.Locale


@Composable
fun NavigationScreen() {

    val context = LocalContext.current


    // ============================================================
    // GPS
    // ============================================================

    var location by remember {
        mutableStateOf<Location?>(null)
    }


    // ============================================================
    // Sensors
    // ============================================================

    var accelerometer by remember {
        mutableStateOf(AccelerometerData())
    }

    var gyroscope by remember {
        mutableStateOf(GyroscopeData())
    }

    var magnetometer by remember {
        mutableStateOf(MagnetometerData())
    }


    // ============================================================
    // Orientation
    // ============================================================

    var heading by remember {
        mutableFloatStateOf(0f)
    }

    var pitch by remember {
        mutableFloatStateOf(0f)
    }

    var roll by remember {
        mutableFloatStateOf(0f)
    }


    // ============================================================
    // GNSS blackout
    // ============================================================

    var blackoutMode by remember {
        mutableStateOf(false)
    }


    // IMPORTANT:
    // Lets the GPS callback always see the latest blackout state.
    val currentBlackoutMode by rememberUpdatedState(blackoutMode)


    // ============================================================
    // Dead Reckoning state
    // ============================================================

    var drState by remember {
        mutableStateOf(
            DeadReckoningState()
        )
    }


    // ============================================================
    // BLACKOUT EVALUATION
    // ============================================================

    // GNSS position at the exact moment blackout starts.
    var blackoutStartLocation by remember {
        mutableStateOf<Location?>(null)
    }

    // Current position error:
    // distance between DR estimated position and GNSS reference.
    var positionError by remember {
        mutableStateOf(0.0)
    }

    // Drift percentage:
    // position error / distance travelled during blackout * 100
    var driftPercentage by remember {
        mutableStateOf(0.0)
    }


    // ============================================================
    // Permission
    // ============================================================

    var permissionGranted by remember {

        mutableStateOf(

            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        )
    }


    // ============================================================
    // Managers
    // ============================================================

    val locationManager =
        remember {
            LocationManager(context)
        }

    val sensorManager =
        remember {
            SensorManager(context)
        }

    val sensorFusionManager =
        remember {
            SensorFusionManager()
        }

    val deadReckoningEngine =
        remember {
            DeadReckoningEngine()
        }

    val productionNavEngine =
        remember {
            ProductionNavigationEngine()
        }

    val highRateSensorManager =
        remember {
            HighRateImuSensorManager(context)
        }

    // Initialize ONNX Runtime session with gru_local.onnx from Android assets
    LaunchedEffect(Unit) {
        productionNavEngine.initializeAssets(context)
        deadReckoningEngine.initializeAssets(context)
    }


    // ============================================================
    // Permission launcher
    // ============================================================

    val permissionLauncher =
        rememberLauncherForActivityResult(

            ActivityResultContracts.RequestPermission()

        ) { granted ->

            permissionGranted = granted
        }


    // ============================================================
    // START GPS + SENSORS
    // ============================================================

    DisposableEffect(permissionGranted) {

        if (permissionGranted) {

            // ----------------------------------------------------
            // GPS
            // ----------------------------------------------------

            locationManager
                .startLocationUpdates {

                        newLocation ->

                    location = newLocation


                    // IMPORTANT:
                    // GPS correction is disabled during blackout.
                    //
                    // Before blackout:
                    // GNSS -> correct DR
                    //
                    // During blackout:
                    // GNSS is only used as reference for evaluation.
                    if (!currentBlackoutMode) {

                        deadReckoningEngine
                            .correctWithGnss(
                                newLocation
                            )

                        productionNavEngine
                            .correctWithGnss(
                                newLocation
                            )

                        val navState =
                            productionNavEngine.getState()

                        drState =
                            DeadReckoningState(
                                latitude = navState.latitude,
                                longitude = navState.longitude,
                                speed = navState.speed,
                                heading = navState.heading,
                                distanceTravelled = navState.distanceTravelled,
                                isInitialized = navState.isInitialized
                            )
                    }
                }


            // ----------------------------------------------------
            // Accelerometer
            // ----------------------------------------------------

            sensorManager
                .startAccelerometer {

                        data ->

                    accelerometer = data

                    sensorFusionManager
                        .updateAccelerometer(
                            data.x,
                            data.y,
                            data.z
                        )
                }


            // ----------------------------------------------------
            // Gyroscope
            // ----------------------------------------------------

            sensorManager
                .startGyroscope {

                        data ->

                    gyroscope = data

                    sensorFusionManager
                        .updateGyroscope(
                            data.x,
                            data.y,
                            data.z
                        )
                }


            // ----------------------------------------------------
            // Magnetometer
            // ----------------------------------------------------

            sensorManager
                .startMagnetometer {

                        data ->

                    magnetometer = data

                    sensorFusionManager
                        .updateMagnetometer(
                            data.x,
                            data.y,
                            data.z
                        )
                }
            // Connect production high-rate IMU pipeline (100 Hz resampler & GRU model)
            productionNavEngine.onNavigationStateChanged = { navState ->
                drState = DeadReckoningState(
                    latitude = navState.latitude,
                    longitude = navState.longitude,
                    speed = navState.speed,
                    heading = navState.heading,
                    distanceTravelled = navState.distanceTravelled,
                    isInitialized = navState.isInitialized
                )
            }

            highRateSensorManager.onImuSampleListener = { sample ->
                productionNavEngine.addSensorSample(sample)
            }

            highRateSensorManager.onOrientationListener = { orientation ->
                productionNavEngine.updateOrientation(orientation)
            }

            highRateSensorManager.start()
        }


        onDispose {

            highRateSensorManager.stop()

            productionNavEngine.close()

            locationManager
                .stopLocationUpdates()

            sensorManager
                .stopAccelerometer()

            sensorManager
                .stopGyroscope()

            sensorManager
                .stopMagnetometer()
        }
    }


    // ============================================================
    // ORIENTATION UPDATE
    // ============================================================

    LaunchedEffect(
        accelerometer,
        magnetometer
    ) {

        val orientation =
            sensorFusionManager
                .getOrientation()

        heading =
            orientation.heading

        pitch =
            orientation.pitch

        roll =
            orientation.roll
    }


    // ============================================================
    // INITIALIZE DEAD RECKONING
    // ============================================================

    LaunchedEffect(location) {

        val currentLocation =
            location

        if (
            currentLocation != null &&
            !drState.isInitialized &&
            !blackoutMode
        ) {

            deadReckoningEngine
                .initialize(
                    currentLocation
                )

            productionNavEngine
                .initialize(
                    currentLocation
                )

            val navState =
                productionNavEngine.getState()

            drState =
                DeadReckoningState(
                    latitude = navState.latitude,
                    longitude = navState.longitude,
                    speed = navState.speed,
                    heading = navState.heading,
                    distanceTravelled = navState.distanceTravelled,
                    isInitialized = navState.isInitialized
                )
        }
    }


    // ============================================================
    // DEAD RECKONING UPDATE
    // ============================================================

    LaunchedEffect(
        accelerometer,
        magnetometer,
        blackoutMode
    ) {

        if (
            blackoutMode &&
            drState.isInitialized
        ) {

            // ----------------------------------------------------
            // Get acceleration in world coordinates
            // ----------------------------------------------------

            val worldAcceleration =
                sensorFusionManager
                    .getWorldAcceleration()


            // ----------------------------------------------------
            // Feed acceleration into DR engine
            // ----------------------------------------------------

            val fallbackState =
                deadReckoningEngine.update(

                    accelerationNorth =
                        worldAcceleration.north,

                    accelerationEast =
                        worldAcceleration.east,

                    heading =
                        heading
                )

            // If productionNavEngine has not yet initialized its model asset, use fallback
            if (!productionNavEngine.modelRunner.ready) {
                drState = fallbackState
            }
        }
    }


    // ============================================================
    // CALCULATE GNSS ERROR + DRIFT
    // ============================================================

    LaunchedEffect(
        location,
        drState,
        blackoutMode
    ) {

        if (
            blackoutMode &&
            location != null &&
            drState.isInitialized
        ) {

            val gpsLocation =
                location!!


            // ----------------------------------------------------
            // Calculate distance between:
            //
            // GNSS reference position
            //             and
            // DR estimated position
            // ----------------------------------------------------

            val results =
                FloatArray(1)

            Location.distanceBetween(

                gpsLocation.latitude,
                gpsLocation.longitude,

                drState.latitude,
                drState.longitude,

                results
            )


            positionError =
                results[0].toDouble()


            // ----------------------------------------------------
            // Calculate drift %
            //
            // Drift = position error / distance travelled * 100
            // ----------------------------------------------------

            val distance =
                drState.distanceTravelled

            driftPercentage =
                if (distance > 1.0) {

                    (positionError / distance) * 100.0

                } else {

                    0.0
                }
        }
    }


    // ============================================================
    // UI
    // ============================================================

    Column(

        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(
                    rememberScrollState()
                )
                .padding(16.dp)
    ) {


        // ========================================================
        // HEADER
        // ========================================================

        Text(

            text = "GUDUMAP",

            style =
                MaterialTheme
                    .typography
                    .headlineMedium
        )


        Text(

            text =
                "Intelligent Dead Reckoning Navigation",

            style =
                MaterialTheme
                    .typography
                    .bodyMedium
        )


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // NAVIGATION STATUS
        // ========================================================

        Card(
            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(
                modifier =
                    Modifier.padding(16.dp)
            ) {

                Text(

                    text =
                        if (blackoutMode)
                            "DEAD RECKONING"
                        else
                            "GNSS NAVIGATION",

                    style =
                        MaterialTheme
                            .typography
                            .titleLarge
                )


                Spacer(
                    modifier =
                        Modifier.height(6.dp)
                )


                Text(

                    text =
                        if (blackoutMode)
                            "GNSS blackout — sensor navigation active"
                        else
                            "GNSS available — GPS navigation active"
                )


                Spacer(
                    modifier =
                        Modifier.height(12.dp)
                )


                Button(

                    modifier =
                        Modifier.fillMaxWidth(),

                    onClick = {

                        blackoutMode =
                            !blackoutMode

                        productionNavEngine
                            .setBlackoutMode(blackoutMode)


                        // ====================================================
                        // START BLACKOUT
                        // ====================================================

                        if (blackoutMode) {

                            location?.let {

                                // Save GNSS position BEFORE blackout.
                                blackoutStartLocation =
                                    Location(it)


                                // Start DR from the current GPS position.
                                deadReckoningEngine
                                    .initialize(it)

                                productionNavEngine
                                    .initialize(it)

                                val navState =
                                    productionNavEngine.getState()

                                drState =
                                    DeadReckoningState(
                                        latitude = navState.latitude,
                                        longitude = navState.longitude,
                                        speed = navState.speed,
                                        heading = navState.heading,
                                        distanceTravelled = navState.distanceTravelled,
                                        isInitialized = navState.isInitialized
                                    )


                                // Reset evaluation values.
                                positionError = 0.0

                                driftPercentage = 0.0
                            }
                        }


                        // ====================================================
                        // END BLACKOUT
                        // ====================================================

                        else {

                            location?.let {

                                // Re-anchor DR with GNSS (smooth EKF recovery).
                                deadReckoningEngine
                                    .correctWithGnss(it)

                                productionNavEngine
                                    .correctWithGnss(it)

                                val navState =
                                    productionNavEngine.getState()

                                drState =
                                    DeadReckoningState(
                                        latitude = navState.latitude,
                                        longitude = navState.longitude,
                                        speed = navState.speed,
                                        heading = navState.heading,
                                        distanceTravelled = navState.distanceTravelled,
                                        isInitialized = navState.isInitialized
                                    )
                            }

                            // Reset blackout evaluation.
                            blackoutStartLocation = null
                            positionError = 0.0
                            driftPercentage = 0.0
                        }
                    }
                ) {

                    Text(

                        text =
                            if (blackoutMode)
                                "END GNSS BLACKOUT"
                            else
                                "START GNSS BLACKOUT"
                    )
                }
            }
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // MAP
        // ========================================================

        MapView(

            location =

                if (
                    blackoutMode &&
                    drState.isInitialized
                ) {

                    Location(
                        "dead_reckoning"
                    ).apply {

                        latitude =
                            drState.latitude

                        longitude =
                            drState.longitude
                    }

                } else {

                    location
                }
        )


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // METRICS
        // ========================================================

        Row(

            modifier =
                Modifier.fillMaxWidth(),

            horizontalArrangement =
                Arrangement.spacedBy(8.dp)
        ) {

            MetricBox(

                title = "Speed",

                value =
                    String.format(
                        Locale.US,
                        "%.1f m/s",
                        drState.speed
                    ),

                modifier =
                    Modifier.weight(1f)
            )


            MetricBox(

                title = "Heading",

                value =
                    String.format(
                        Locale.US,
                        "%.0f°",
                        heading
                    ),

                modifier =
                    Modifier.weight(1f)
            )


            MetricBox(

                title = "Distance",

                value =
                    String.format(
                        Locale.US,
                        "%.1f m",
                        drState.distanceTravelled
                    ),

                modifier =
                    Modifier.weight(1f)
            )
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // BLACKOUT PERFORMANCE
        // ========================================================

        Card(

            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(

                modifier =
                    Modifier.padding(16.dp)
            ) {

                Text(

                    text =
                        "Blackout Performance",

                    style =
                        MaterialTheme
                            .typography
                            .titleMedium
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(

                    text =
                        if (blackoutMode)
                            "Evaluation active"
                        else
                            "Start GNSS blackout to evaluate DR"
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                // ------------------------------------------------
                // GNSS reference
                // ------------------------------------------------

                if (blackoutStartLocation != null) {

                    Text(

                        text =
                            String.format(
                                Locale.US,
                                "Blackout Start: %.6f, %.6f",
                                blackoutStartLocation!!.latitude,
                                blackoutStartLocation!!.longitude
                            )
                    )
                }


                // ------------------------------------------------
                // Current GNSS ground truth
                // ------------------------------------------------

                if (location != null) {

                    Text(

                        text =
                            String.format(
                                Locale.US,
                                "GNSS Reference: %.6f, %.6f",
                                location!!.latitude,
                                location!!.longitude
                            )
                    )
                }


                // ------------------------------------------------
                // DR estimated position
                // ------------------------------------------------

                Text(

                    text =
                        String.format(
                            Locale.US,
                            "DR Estimate: %.6f, %.6f",
                            drState.latitude,
                            drState.longitude
                        )
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                // ------------------------------------------------
                // Position error
                // ------------------------------------------------

                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Position Error: %.2f m",
                            positionError
                        )
                )


                // ------------------------------------------------
                // Distance travelled
                // ------------------------------------------------

                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Distance Travelled: %.2f m",
                            drState.distanceTravelled
                        )
                )


                // ------------------------------------------------
                // Drift
                // ------------------------------------------------

                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Drift: %.2f%%",
                            driftPercentage
                        )
                )
            }
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // ESTIMATED POSITION
        // ========================================================

        Card(

            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(

                modifier =
                    Modifier.padding(16.dp)
            ) {

                Text(

                    text =
                        "Estimated Position",

                    style =
                        MaterialTheme
                            .typography
                            .titleMedium
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Latitude: %.6f",
                            drState.latitude
                        )
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Longitude: %.6f",
                            drState.longitude
                        )
                )
            }
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // ORIENTATION
        // ========================================================

        Card(

            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(

                modifier =
                    Modifier.padding(16.dp)
            ) {

                Text(

                    text =
                        "Phone Orientation",

                    style =
                        MaterialTheme
                            .typography
                            .titleMedium
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Heading: %.1f°",
                            heading
                        )
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Pitch: %.1f°",
                            pitch
                        )
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Roll: %.1f°",
                            roll
                        )
                )
            }
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // SENSOR DATA
        // ========================================================

        Card(

            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(

                modifier =
                    Modifier.padding(16.dp)
            ) {

                Text(

                    text =
                        "Sensor Data",

                    style =
                        MaterialTheme
                            .typography
                            .titleMedium
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Accelerometer: %.2f m/s²",
                            accelerometer.magnitude
                        )
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "X: %.2f",
                            accelerometer.x
                        )
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Y: %.2f",
                            accelerometer.y
                        )
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Z: %.2f",
                            accelerometer.z
                        )
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Gyroscope: %.2f rad/s",
                            gyroscope.magnitude
                        )
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(

                    text =
                        String.format(
                            Locale.US,
                            "Magnetometer: %.2f µT",
                            magnetometer.magnitude
                        )
                )
            }
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // GNSS STATUS
        // ========================================================

        Card(

            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(

                modifier =
                    Modifier.padding(16.dp)
            ) {

                Text(

                    text =
                        "GNSS Status",

                    style =
                        MaterialTheme
                            .typography
                            .titleMedium
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(

                    text =
                        if (location != null)
                            "GPS location received"
                        else
                            "Waiting for GPS..."
                )


                if (location != null) {

                    Text(

                        text =
                            String.format(
                                Locale.US,
                                "GPS Latitude: %.6f",
                                location!!.latitude
                            )
                    )


                    Text(

                        text =
                            String.format(
                                Locale.US,
                                "GPS Longitude: %.6f",
                                location!!.longitude
                            )
                    )


                    if (location!!.hasSpeed()) {

                        Text(

                            text =
                                String.format(
                                    Locale.US,
                                    "GPS Speed: %.2f m/s",
                                    location!!.speed
                                )
                        )
                    }
                }


                Spacer(
                    modifier =
                        Modifier.height(6.dp)
                )


                Text(

                    text =
                        if (blackoutMode)
                            "GPS is NOT correcting Dead Reckoning"
                        else
                            "GPS correction is ACTIVE"
                )
            }
        }


        Spacer(
            modifier =
                Modifier.height(12.dp)
        )


        // ========================================================
        // LOCATION PERMISSION
        // ========================================================

        if (!permissionGranted) {

            Button(

                modifier =
                    Modifier.fillMaxWidth(),

                onClick = {

                    permissionLauncher.launch(
                        Manifest.permission.ACCESS_FINE_LOCATION
                    )
                }
            ) {

                Text(
                    text =
                        "ALLOW LOCATION"
                )
            }
        }


        Spacer(
            modifier =
                Modifier.height(30.dp)
        )
    }
}


// ================================================================
// METRIC BOX
// ================================================================

@Composable
private fun MetricBox(

    title: String,

    value: String,

    modifier: Modifier = Modifier

) {

    Card(

        modifier =
            modifier
    ) {

        Column(

            modifier =
                Modifier.padding(10.dp),

            horizontalAlignment =
                Alignment.CenterHorizontally
        ) {

            Text(

                text =
                    title,

                style =
                    MaterialTheme
                        .typography
                        .labelMedium
            )


            Text(

                text =
                    value,

                style =
                    MaterialTheme
                        .typography
                        .titleMedium
            )
        }
    }
}