import org.gradle.process.CommandLineArgumentProvider

plugins {
    id("com.android.application") version "9.5.2"
}

val mobile = rootProject.projectDir.parentFile
val studio = mobile.parentFile
val remeVersion = Regex("""(?m)^__version__ = "(\d+\.\d+\.\d+)"$""")
    .find(studio.resolve("../reme/_version.py").readText())!!.groupValues[1]
val (major, minor, patch) = remeVersion.split(".").map(String::toInt)
require(major < 2100 && minor < 1000 && patch < 1000) { "ReMe version exceeds the Android version code range" }

abstract class BundleStudio : Exec() {
    @get:OutputDirectory
    abstract val outputDirectory: DirectoryProperty
}

android {
    namespace = "io.agentscope.reme.studio"
    compileSdk { version = release(37) { minorApiLevel = 2 } }

    defaultConfig {
        applicationId = "io.agentscope.reme.studio"
        minSdk = 29
        targetSdk = 37
        versionCode = major * 1_000_000 + minor * 1_000 + patch + 1
        versionName = remeVersion
    }

    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

androidComponents {
    onVariants { variant ->
        val name = variant.name.replaceFirstChar { it.uppercaseChar() }
        val bundle = tasks.register<BundleStudio>("bundle${name}Studio") {
            workingDir(mobile)
            commandLine("node", studio.resolve("node_modules/vite/bin/vite.js"), "build", "--emptyOutDir")
            inputs.files(
                mobile.resolve("index.html"),
                mobile.resolve("package.json"),
                mobile.resolve("vite.config.mts"),
                studio.resolve("package-lock.json"),
            )
            inputs.files(
                mobile.resolve("src"),
                studio.resolve("client/src"),
                studio.resolve("workbench/src"),
                studio.resolve("workbench/public"),
            )
            inputs.file(studio.resolve("../reme/_version.py"))
            argumentProviders.add(CommandLineArgumentProvider { listOf("--outDir", outputDirectory.get().asFile.absolutePath) })
        }
        variant.sources.assets?.addGeneratedSourceDirectory(bundle, BundleStudio::outputDirectory)
    }
}

dependencies {
    implementation("androidx.activity:activity:1.13.0")
    implementation("androidx.webkit:webkit:1.18.0")
}
