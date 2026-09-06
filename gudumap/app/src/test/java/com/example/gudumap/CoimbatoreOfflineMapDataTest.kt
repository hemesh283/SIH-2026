package com.example.gudumap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FileInputStream

class CoimbatoreOfflineMapDataTest {

    @Test
    fun testMBTilesDatabaseFileHeaderAndSize() {
        val mbtilesFile = File("src/main/assets/maps/coimbatore/coimbatore.mbtiles")
        val altFile = File("app/src/main/assets/maps/coimbatore/coimbatore.mbtiles")
        val target = if (mbtilesFile.exists()) mbtilesFile else altFile

        assertTrue("MBTiles file must exist in assets", target.exists())
        assertTrue("MBTiles file size must be > 10 MB (real full dataset)", target.length() > 10 * 1024 * 1024)

        // Read SQLite 3 16-byte magic header
        val header = ByteArray(16)
        FileInputStream(target).use { it.read(header) }
        val magic = String(header, 0, 15, Charsets.US_ASCII)
        assertEquals("SQLite format 3", magic)
    }

    @Test
    fun testCoimbatoreRoadsJsonIntegrity() {
        val jsonFile = File("src/main/assets/maps/coimbatore/coimbatore_roads.json")
        val altFile = File("app/src/main/assets/maps/coimbatore/coimbatore_roads.json")
        val target = if (jsonFile.exists()) jsonFile else altFile

        assertTrue("Roads JSON must exist", target.exists())
        val text = target.readText()

        // Verify JSON contains expected root structure and roads array
        assertTrue(text.contains("\"roads\":"))
        assertTrue("Must contain Avinashi Road", text.contains("Avinashi", ignoreCase = true))
        assertTrue("Must contain Trichy Road", text.contains("Trichy", ignoreCase = true))
        assertTrue("Must contain Mettupalayam Road", text.contains("Mettupalayam", ignoreCase = true) || text.contains("MTP", ignoreCase = true))
        assertTrue("Must contain Sathy Road", text.contains("Sathy", ignoreCase = true))
        assertTrue("Must contain Pollachi Road", text.contains("Pollachi", ignoreCase = true))
        assertTrue("Must contain DB Road", text.contains("DB Road", ignoreCase = true) || text.contains("Diwan Bahadur", ignoreCase = true))
    }
}
