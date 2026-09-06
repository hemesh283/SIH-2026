package com.example.gudumap.sensors

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager as AndroidLocationManager
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat

class LocationManager(
    private val context: Context
) {

    companion object {
        private const val TAG = "Gudumap:LocationMgr"
    }

    private val locationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as AndroidLocationManager

    private var listener: LocationListener? = null

    var isGnssAvailable: Boolean = false
        private set

    /**
     * True once [startLocationUpdates] has successfully registered a listener. Lets callers
     * (e.g. a permission-grant callback, or an onResume retry) check whether a retry is even
     * useful before calling again, and lets [startLocationUpdates] itself stay safe to call
     * more than once (PROJECT_STATUS.md §14 -- the fix for the registration-timing gap in §13).
     */
    fun isListening(): Boolean = listener != null

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
        onLocationChanged: (Location) -> Unit,
        onStatusChanged: ((Boolean) -> Unit)? = null
    ) {
        // Safe to call more than once (Fix, PROJECT_STATUS.md §14): if a listener is already
        // registered, tear it down cleanly first rather than accumulating a second one. This is
        // what makes retrying after a late permission grant (or an onResume recheck) safe no
        // matter how many times it happens.
        if (listener != null) {
            Log.i(TAG, "startLocationUpdates: already listening -- re-registering cleanly")
            stopLocationUpdates()
        }

        // TEMPORARY DIAGNOSTIC LOGGING (PROJECT_STATUS.md §13) -- this is the exact branch point
        // that determines whether Android is ever even asked for location updates at all.
        if (!hasLocationPermission()) {
            Log.w(TAG, "startLocationUpdates: permission NOT granted at this moment -- " +
                "requestLocationUpdates will NEVER be called unless start() runs again later")
            isGnssAvailable = false
            onStatusChanged?.invoke(false)
            return
        }
        Log.i(TAG, "startLocationUpdates: permission granted, registering listeners")

        listener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                Log.i(TAG, "onLocationChanged: provider=${location.provider} " +
                    "lat=${location.latitude} lon=${location.longitude} " +
                    "accuracy=${location.accuracy}m hasSpeed=${location.hasSpeed()}")
                isGnssAvailable = true
                onStatusChanged?.invoke(true)
                onLocationChanged(location)
            }

            override fun onProviderEnabled(provider: String) {
                if (provider == AndroidLocationManager.GPS_PROVIDER) {
                    isGnssAvailable = true
                    onStatusChanged?.invoke(true)
                }
            }

            override fun onProviderDisabled(provider: String) {
                if (provider == AndroidLocationManager.GPS_PROVIDER) {
                    isGnssAvailable = false
                    onStatusChanged?.invoke(false)
                }
            }

            @Deprecated("Deprecated in Java")
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {
                // Compatibility for older APIs
            }
        }

        // Try GPS provider
        val gpsEnabled = locationManager.isProviderEnabled(AndroidLocationManager.GPS_PROVIDER)
        Log.i(TAG, "GPS_PROVIDER enabled=$gpsEnabled")
        if (gpsEnabled) {
            locationManager.requestLocationUpdates(
                AndroidLocationManager.GPS_PROVIDER,
                1000L,
                1f,
                listener!!
            )
            isGnssAvailable = true
        }

        // Fallback to Network provider
        val networkEnabled = locationManager.isProviderEnabled(AndroidLocationManager.NETWORK_PROVIDER)
        Log.i(TAG, "NETWORK_PROVIDER enabled=$networkEnabled")
        if (networkEnabled) {
            locationManager.requestLocationUpdates(
                AndroidLocationManager.NETWORK_PROVIDER,
                1000L,
                1f,
                listener!!
            )
        }

        onStatusChanged?.invoke(isGnssAvailable)
    }

    fun stopLocationUpdates() {
        listener?.let {
            locationManager.removeUpdates(it)
        }
        listener = null
        isGnssAvailable = false
    }
}