package com.example.gudumap

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge

import com.example.gudumap.ui.screens.NavigationScreen
import com.example.gudumap.ui.theme.GudumapTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContent {

            GudumapTheme {

                NavigationScreen()

            }
        }
    }
}