plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.gudumap"

    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.gudumap"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        compose = true
    }

    androidResources {
        noCompress += listOf("onnx", "map", "mbtiles", "sqlite", "json")
    }
}

dependencies {

    // ============================================================
    // Jetpack Compose
    // ============================================================

    implementation(platform(libs.androidx.compose.bom))

    implementation(libs.androidx.activity.compose)

    implementation(libs.androidx.compose.material3)

    implementation(libs.androidx.compose.ui)

    implementation(libs.androidx.compose.ui.graphics)

    implementation(libs.androidx.compose.ui.tooling.preview)


    // ============================================================
    // AndroidX & Architecture
    // ============================================================

    implementation(libs.androidx.core.ktx)

    implementation(libs.androidx.lifecycle.runtime.ktx)

    implementation(libs.androidx.lifecycle.viewmodel.ktx)

    implementation(libs.androidx.lifecycle.viewmodel.compose)


    // ============================================================
    // Machine Learning Runtime (ONNX Runtime)
    // ============================================================

    implementation(libs.onnxruntime.android)


    // ============================================================
    // OpenStreetMap / osmdroid
    // ============================================================

    implementation(libs.osmdroid.android)


    // ============================================================
    // Google Fused Location Provider
    // Used for real-time phone GPS location
    // ============================================================

    implementation("com.google.android.gms:play-services-location:21.3.0")


    // ============================================================
    // Testing
    // ============================================================

    testImplementation(libs.junit)
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.17.0")

    androidTestImplementation(
        platform(libs.androidx.compose.bom)
    )

    androidTestImplementation(
        libs.androidx.compose.ui.test.junit4
    )

    androidTestImplementation(
        libs.androidx.espresso.core
    )

    androidTestImplementation(
        libs.androidx.junit
    )


    // ============================================================
    // Debug
    // ============================================================

    debugImplementation(
        libs.androidx.compose.ui.test.manifest
    )

    debugImplementation(
        libs.androidx.compose.ui.tooling
    )
}