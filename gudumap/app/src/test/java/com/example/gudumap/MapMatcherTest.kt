package com.example.gudumap

import com.example.gudumap.map.MapMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class MapMatcherTest {

    @Test
    fun testEmptyRoadsGracefulFallback() {
        val matcher = MapMatcher(null)
        val result = matcher.match(
            estimatedLat = 11.0168,
            estimatedLon = 76.9558,
            headingDeg = 90f,
            speedKmh = 25f
        )

        assertFalse(result.isSnapped)
        assertEquals(11.0168, result.matchedLat, 1e-6)
        assertEquals(76.9558, result.matchedLon, 1e-6)
    }
}
