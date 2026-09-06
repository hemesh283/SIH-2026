package com.example.gudumap.sensors

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager as AndroidLocationManager

import androidx.core.content.ContextCompat


class LocationManager(
    private val context: Context
) {

    private val locationManager =
        context.getSystemService(
            Context.LOCATION_SERVICE
        ) as AndroidLocationManager

    private var listener: LocationListener? = null


    fun hasLocationPermission(): Boolean {

        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED ||

                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED
    }


    @SuppressLint("MissingPermission")
    fun startLocationUpdates(
        onLocationChanged: (Location) -> Unit
    ) {

        if (!hasLocationPermission()) {
            return
        }


        listener = object : LocationListener {

            override fun onLocationChanged(
                location: Location
            ) {
                onLocationChanged(location)
            }

            override fun onProviderEnabled(
                provider: String
            ) {
            }

            override fun onProviderDisabled(
                provider: String
            ) {
            }
        }


        // GPS
        if (
            locationManager.isProviderEnabled(
                AndroidLocationManager.GPS_PROVIDER
            )
        ) {

            locationManager.requestLocationUpdates(
                AndroidLocationManager.GPS_PROVIDER,
                1000L,
                1f,
                listener!!
            )
        }


        // Network location
        if (
            locationManager.isProviderEnabled(
                AndroidLocationManager.NETWORK_PROVIDER
            )
        ) {

            locationManager.requestLocationUpdates(
                AndroidLocationManager.NETWORK_PROVIDER,
                1000L,
                1f,
                listener!!
            )
        }
    }


    fun stopLocationUpdates() {

        listener?.let {

            locationManager.removeUpdates(it)
        }

        listener = null
    }
}