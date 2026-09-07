package com.example.gudumap.map

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.MapTileProviderArray
import org.osmdroid.tileprovider.modules.ArchiveFileFactory
import org.osmdroid.tileprovider.modules.IArchiveFile
import org.osmdroid.tileprovider.modules.MBTilesFileArchive
import org.osmdroid.tileprovider.modules.MapTileFileArchiveProvider
import org.osmdroid.tileprovider.modules.MapTileModuleProviderBase
import org.osmdroid.tileprovider.tilesource.ITileSource
import org.osmdroid.tileprovider.tilesource.XYTileSource
import org.osmdroid.tileprovider.util.SimpleRegisterReceiver
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import java.io.File
import java.io.FileOutputStream

enum class OfflineMapStatus {
    AVAILABLE,
    LOADING,
    ERROR,
    NOT_AVAILABLE
}

class OfflineMapManager(private val context: Context) {

    companion object {
        private const val TAG = "Gudumap:OfflineMap"
        const val COIMBATORE_DEFAULT_LAT = 11.0168
        const val COIMBATORE_DEFAULT_LON = 76.9558

        // §22: MUST match coimbatore.mbtiles' own `metadata` table (minzoom/maxzoom rows) exactly.
        // Verified directly against the bundled file via sqlite3 -- real data covers zoom 11-16
        // only (915 tiles: 11=4, 12=12, 13=30, 14=110, 15=399, 16=360; metadata row maxzoom=16).
        // This constant was previously 17 -- one level past the last real tile -- which let the
        // map UI zoom into a range with zero data (see MapView.kt's zoom bounds, now derived from
        // this same constant instead of an independently hardcoded 18.0).
        const val MIN_ZOOM = 11
        const val MAX_ZOOM = 16

        // Bounding box for Coimbatore metropolitan area (23.3 km x 20.7 km, ~482 sq km)
        val COIMBATORE_BOUNDS = BoundingBox(11.125, 77.070, 10.915, 76.880)
        val COIMBATORE_CENTER = GeoPoint(COIMBATORE_DEFAULT_LAT, COIMBATORE_DEFAULT_LON)
    }

    var status: OfflineMapStatus = OfflineMapStatus.LOADING
        private set

    private var localMapFile: File? = null
    private var tileCount: Int = 0

    init {
        configureOsmdroid()
        initializeOfflineMap()
    }

    private fun configureOsmdroid() {
        try {
            Configuration.getInstance().load(
                context,
                context.getSharedPreferences("gudumap_osmdroid_prefs", Context.MODE_PRIVATE)
            )
            Configuration.getInstance().userAgentValue = "Gudumap/1.0 (Android; Offline-Coimbatore)"
        } catch (e: Exception) {
            Log.w(TAG, "Failed to configure osmdroid: ${e.message}")
        }
    }

    /**
     * Initializes the offline map dataset:
     * 1. Locates or creates the internal storage directory: context.filesDir/maps/coimbatore/
     * 2. Copies coimbatore.mbtiles from APK assets on first run or when updated
     * 3. Verifies SQLite integrity and tile counts
     * 4. Updates status to AVAILABLE, ERROR, or NOT_AVAILABLE
     */
    @Synchronized
    fun initializeOfflineMap() {
        status = OfflineMapStatus.LOADING
        try {
            val mapsDir = File(context.filesDir, "maps/coimbatore")
            if (!mapsDir.exists()) mapsDir.mkdirs()

            val targetFile = File(mapsDir, "coimbatore.mbtiles")

            // Determine if asset exists
            val assetPath = try {
                context.assets.open("maps/coimbatore/coimbatore.mbtiles").close()
                "maps/coimbatore/coimbatore.mbtiles"
            } catch (e: Exception) {
                try {
                    context.assets.open("maps/coimbatore.mbtiles").close()
                    "maps/coimbatore.mbtiles"
                } catch (e2: Exception) {
                    null
                }
            }

            if (assetPath == null && !targetFile.exists()) {
                Log.e(TAG, "Offline map asset not found in assets/maps/")
                status = OfflineMapStatus.NOT_AVAILABLE
                return
            }

            // §26: re-copy if the bundled asset's size differs from what's already sitting in
            // internal storage, not just when the target is missing/empty. Merged back from
            // teammate's branch -- without this, a future APK update that ships a
            // corrected/updated coimbatore.mbtiles would silently keep using whatever was
            // copied on first install forever (internal storage is never touched again once a
            // non-empty file exists there). Only matters for a future map-data update; has no
            // effect on a fresh install, where targetFile doesn't exist yet either way.
            val assetSize = if (assetPath != null) {
                try {
                    context.assets.openFd(assetPath).length
                } catch (e: Exception) {
                    try {
                        context.assets.open(assetPath).use { it.available().toLong() }
                    } catch (e2: Exception) {
                        0L
                    }
                }
            } else 0L

            val shouldCopy = assetPath != null && (
                !targetFile.exists() ||
                targetFile.length() == 0L ||
                (assetSize > 0L && targetFile.length() != assetSize)
            )

            if (shouldCopy && assetPath != null) {
                Log.i(TAG, "Copying offline Coimbatore map package from $assetPath to ${targetFile.absolutePath} (asset size: ${assetSize} B)...")
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(targetFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "Extracted offline map successfully (${targetFile.length() / (1024 * 1024)} MB).")
            }

            localMapFile = targetFile

            // Verify SQLite integrity
            if (targetFile.exists() && targetFile.length() > 0L) {
                val isValid = verifyDatabase(targetFile)
                if (isValid) {
                    status = OfflineMapStatus.AVAILABLE
                    Log.i(TAG, "Offline Coimbatore map is AVAILABLE. Tile count = $tileCount, size = ${targetFile.length() / (1024 * 1024)} MB.")
                } else {
                    status = OfflineMapStatus.ERROR
                    Log.e(TAG, "Offline map file exists but verification failed.")
                }
            } else {
                status = OfflineMapStatus.NOT_AVAILABLE
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing offline map: ${e.message}", e)
            status = OfflineMapStatus.ERROR
        }
    }

    private fun verifyDatabase(file: File): Boolean {
        var db: SQLiteDatabase? = null
        return try {
            db = SQLiteDatabase.openDatabase(file.absolutePath, null, SQLiteDatabase.OPEN_READONLY)
            val cursor = db.rawQuery("SELECT COUNT(*) FROM tiles", null)
            var count = 0
            if (cursor.moveToFirst()) {
                count = cursor.getInt(0)
            }
            cursor.close()
            tileCount = count
            count > 0
        } catch (e: Exception) {
            Log.e(TAG, "SQLite validation error: ${e.message}", e)
            false
        } finally {
            try { db?.close() } catch (_: Exception) {}
        }
    }

    fun getLocalMapFile(): File? = localMapFile

    fun getTileCount(): Int = tileCount

    fun isOfflineMapAvailable(): Boolean = status == OfflineMapStatus.AVAILABLE

    /**
     * Creates an offline-only MapTileProviderArray backed by the local MBTiles archive.
     * This provider contains NO network downloader modules and enforces offline rendering.
     */
    fun createOfflineTileProvider(): MapTileProviderArray? {
        val file = localMapFile ?: run {
            Log.e(TAG, "createOfflineTileProvider: no local map file recorded (initializeOfflineMap() never reached copy step)")
            return null
        }
        if (!file.exists()) {
            Log.e(TAG, "createOfflineTileProvider: local map file does not exist at ${file.absolutePath}")
            return null
        }
        Log.i(TAG, "createOfflineTileProvider: mbtiles file found at ${file.absolutePath} (${file.length() / 1024}KB)")

        return try {
            val archive: IArchiveFile? = try {
                MBTilesFileArchive.getDatabaseFileArchive(file)
            } catch (e: Exception) {
                Log.w(TAG, "MBTilesFileArchive.getDatabaseFileArchive failed (${e.message}), trying ArchiveFileFactory fallback")
                ArchiveFileFactory.getArchiveFile(file)
            }

            if (archive == null) {
                Log.e(TAG, "createOfflineTileProvider: could not open ${file.absolutePath} as an MBTiles archive -- both MBTilesFileArchive and ArchiveFileFactory failed")
                return null
            }
            Log.i(TAG, "createOfflineTileProvider: MBTiles archive opened successfully ($archive)")

            archive.setIgnoreTileSource(true)

            val offlineTileSource: ITileSource = XYTileSource(
                "CoimbatoreOffline",
                MIN_ZOOM,
                MAX_ZOOM,
                256,
                ".png",
                emptyArray()
            )

            val receiver = SimpleRegisterReceiver(context)
            val archiveProvider = MapTileFileArchiveProvider(
                receiver,
                offlineTileSource,
                arrayOf(archive),
                true
            )

            Log.i(TAG, "createOfflineTileProvider: ready -- tileCount=$tileCount (from verifyDatabase), zoom range [$MIN_ZOOM, $MAX_ZOOM]")

            MapTileProviderArray(
                offlineTileSource,
                receiver,
                arrayOf<MapTileModuleProviderBase>(archiveProvider)
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error creating offline tile provider: ${e.message}", e)
            null
        }
    }

    fun isOnline(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return false
        val network = cm.activeNetwork ?: return false
        val capabilities = cm.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    fun getMapStatus(blackoutMode: Boolean = true): String {
        return "OFFLINE"
    }

    fun getOfflineMapStatusString(): String {
        return when (status) {
            OfflineMapStatus.AVAILABLE -> "AVAILABLE"
            OfflineMapStatus.LOADING -> "LOADING"
            OfflineMapStatus.ERROR -> "ERROR"
            OfflineMapStatus.NOT_AVAILABLE -> "NOT_AVAILABLE"
        }
    }
}
