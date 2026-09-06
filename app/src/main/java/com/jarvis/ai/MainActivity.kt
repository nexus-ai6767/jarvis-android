package com.jarvis.ai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin

private const val RECORD_AUDIO_REQUEST = 501
private const val MODEL_NAME = "gemma-2b-it-cpu-int4.bin"
private const val MODEL_URL = "https://storage.googleapis.com/jmstore/kaggleweb/grader/g-2b-it-cpu-int4.bin"
private const val MODEL_PREFS = "jarvis_local_ai"
private const val MODEL_READY = "model_ready"

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    private var speechRecognizer: SpeechRecognizer? = null
    private lateinit var tts: TextToSpeech
    private var ttsReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppContextHolder.context = applicationContext
        tts = TextToSpeech(this, this)
        setupSpeechRecognizer()
        setContent {
            JarvisApp(
                modelFile = File(filesDir, MODEL_NAME),
                onSpeak = { speak(it) },
                onListen = { startListening() }
            )
        }
    }

    override fun onInit(status: Int) {
        ttsReady = status == TextToSpeech.SUCCESS
        if (ttsReady) {
            tts.language = Locale.UK
            tts.setSpeechRate(0.92f)
            tts.setPitch(0.88f)
        }
    }

    private fun speak(text: String) {
        if (!ttsReady || text.isBlank()) return
        tts.speak(text.take(3000), TextToSpeech.QUEUE_FLUSH, null, "jarvis")
    }

    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) return
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Voice recognition stopped", Toast.LENGTH_SHORT).show()
                }
            }
            override fun onResults(results: Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull().orEmpty()
                if (text.isNotBlank()) runOnUiThread { VoiceBus.emit(text) }
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun startListening() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_AUDIO_REQUEST)
            return
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
        }
        speechRecognizer?.startListening(intent)
    }

    override fun onDestroy() {
        speechRecognizer?.destroy()
        if (ttsReady) tts.stop()
        tts.shutdown()
        super.onDestroy()
    }
}

private object VoiceBus {
    private val listeners = mutableListOf<(String) -> Unit>()
    fun subscribe(listener: (String) -> Unit) { listeners += listener }
    fun unsubscribe(listener: (String) -> Unit) { listeners -= listener }
    fun emit(text: String) { listeners.toList().forEach { it(text) } }
}

data class ChatMessage(val text: String, val fromUser: Boolean)

@Composable
fun JarvisApp(
    modelFile: File,
    onSpeak: (String) -> Unit,
    onListen: () -> Unit
) {
    var input by remember { mutableStateOf("") }
    var thinking by remember { mutableStateOf(false) }
    var modelReady by remember { mutableStateOf(modelFile.exists() && modelFile.length() > 100_000_000L) }
    var downloading by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableStateOf(0) }
    var showInfo by remember { mutableStateOf(false) }
    var localEngine by remember { mutableStateOf<LlmEngine?>(null) }

    val messages = remember {
        mutableStateListOf(
            ChatMessage(
                if (modelReady) "Local AI core detected. JARVIS is ready for offline inference." else "Systems online. Local AI core is not installed yet.",
                false
            )
        )
    }
    val listState = rememberLazyListState()

    fun downloadModel() {
        if (downloading || modelReady) return
        downloading = true
        downloadProgress = 0
        messages += ChatMessage("Downloading local AI core. This is a one-time setup.", false)
        Thread {
            try {
                val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 15000
                    readTimeout = 60000
                    requestMethod = "GET"
                }
                connection.connect()
                if (connection.responseCode !in 200..299) throw IllegalStateException("Model download failed: HTTP ${connection.responseCode}")
                val total = connection.contentLengthLong
                val temp = File(modelFile.parentFile, "$MODEL_NAME.download")
                connection.inputStream.use { inputStream ->
                    temp.outputStream().use { outputStream ->
                        val buffer = ByteArray(1024 * 1024)
                        var copied = 0L
                        while (true) {
                            val read = inputStream.read(buffer)
                            if (read < 0) break
                            outputStream.write(buffer, 0, read)
                            copied += read
                            if (total > 0) {
                                val percent = ((copied * 100L) / total).toInt().coerceIn(0, 100)
                                android.os.Handler(android.os.Looper.getMainLooper()).post { downloadProgress = percent }
                            }
                        }
                    }
                }
                connection.disconnect()
                if (temp.length() < 100_000_000L) throw IllegalStateException("Downloaded model file is incomplete.")
                if (!temp.renameTo(modelFile)) {
                    temp.copyTo(modelFile, overwrite = true)
                    temp.delete()
                }
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    modelReady = true
                    downloading = false
                    downloadProgress = 100
                    messages += ChatMessage("Local AI core installed. Internet is no longer required for AI responses.", false)
                }
            } catch (e: Exception) {
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    downloading = false
                    messages += ChatMessage("Model setup failed: ${e.message ?: "unknown error"}", false)
                }
            }
        }.start()
    }

    fun ask(text: String) {
        val clean = text.trim()
        if (clean.isBlank() || thinking) return
        messages += ChatMessage(clean, true)
        input = ""
        thinking = true
        Thread {
            val result = try {
                if (!modelReady) {
                    "The local AI core is not installed yet. Use INSTALL LOCAL AI once while connected to the internet."
                } else {
                    if (localEngine == null) localEngine = LlmEngine(modelFile)
                    localEngine!!.ask(buildPrompt(messages.takeLast(12)))
                }
            } catch (e: Exception) {
                "Local AI error: ${e.message ?: "inference failed"}"
            }
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                messages += ChatMessage(result.ifBlank { "I couldn't generate a response." }, false)
                thinking = false
                onSpeak(result)
            }
        }.start()
    }

    DisposableEffect(Unit) {
        val listener: (String) -> Unit = { ask(it) }
        VoiceBus.subscribe(listener)
        onDispose {
            VoiceBus.unsubscribe(listener)
            localEngine?.close()
        }
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex)
    }

    val transition = rememberInfiniteTransition(label = "reactor")
    val rotation by transition.animateFloat(0f, 360f, infiniteRepeatable(tween(9000, easing = LinearEasing)), label = "rotation")
    val pulse by transition.animateFloat(0.94f, 1.06f, infiniteRepeatable(tween(1200), RepeatMode.Reverse), label = "pulse")

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        Canvas(Modifier.fillMaxSize()) {
            val cx = size.width / 2f
            val cy = size.height * 0.27f
            val r = minOf(size.width, size.height) * 0.19f * pulse
            drawCircle(
                brush = Brush.radialGradient(
                    listOf(Color(0x66FF8A00), Color(0x22FF6500), Color.Transparent),
                    Offset(cx, cy), r * 2.2f
                ),
                radius = r * 2.2f,
                center = Offset(cx, cy)
            )
            for (i in 0..4) {
                val rr = r * (0.75f + i * 0.14f)
                drawOval(
                    color = Color(0x66FF8A00),
                    topLeft = Offset(cx - rr, cy - rr * 0.34f),
                    size = androidx.compose.ui.geometry.Size(rr * 2f, rr * 0.68f),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = if (i == 2) 2.2f else 1f)
                )
            }
            for (i in 0 until 54) {
                val a = (i * 6.67f + rotation) * Math.PI / 180.0
                val x = cos(a).toFloat()
                val y = sin(a * 2.1).toFloat()
                drawLine(
                    color = Color(0xAAFF7800),
                    start = Offset(cx - x * r * 0.45f, cy - y * r * 0.45f),
                    end = Offset(cx + x * r, cy + y * r),
                    strokeWidth = 1.05f
                )
            }
            drawCircle(
                brush = Brush.radialGradient(listOf(Color(0xFFFFF1C7), Color(0xFFFF8A00), Color.Transparent)),
                radius = r * 0.42f,
                center = Offset(cx, cy)
            )
            drawCircle(Color(0xFFFFB347), r * 0.14f, Offset(cx, cy))
        }

        Column(Modifier.fillMaxSize().padding(18.dp)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("J.A.R.V.I.S", color = Color(0xFFFF8A00), fontSize = 22.sp, fontWeight = FontWeight.Bold)
                    Text("ANDROID // OFFLINE AI", color = Color(0x99FF8A00), fontSize = 9.sp)
                }
                TextButton(onClick = { showInfo = !showInfo }) {
                    Text("LOCAL AI", color = Color(0xFFFFA033), fontSize = 10.sp)
                }
            }

            if (showInfo) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xCC151515)),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Column(Modifier.padding(14.dp)) {
                        Text("ON-DEVICE AI", color = Color(0xFFFFA033), fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(6.dp))
                        Text(
                            if (modelReady) "Gemma 2B local model installed. AI answers run on this phone." else "A local Gemma model must be downloaded once. After installation, AI inference runs on the phone.",
                            color = Color.LightGray,
                            fontSize = 10.sp
                        )
                        Spacer(Modifier.height(8.dp))
                        if (!modelReady) {
                            Button(
                                onClick = { downloadModel() },
                                enabled = !downloading,
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF9A4D00)),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(if (downloading) "DOWNLOADING $downloadProgress%" else "INSTALL LOCAL AI")
                            }
                        } else {
                            Text("● LOCAL AI READY", color = Color(0xFFFFB347), fontSize = 10.sp, fontWeight = FontWeight.Bold)
                        }
                        Spacer(Modifier.height(4.dp))
                        Text("The model is stored in this app's private storage. The app does not need a Gemini API key.", color = Color.Gray, fontSize = 9.sp)
                    }
                }
            }

            Spacer(Modifier.height(150.dp))
            Text("SYSTEM ONLINE", color = Color(0xFFFFA033), fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.CenterHorizontally))
            Text(
                when {
                    downloading -> "INSTALLING LOCAL AI..."
                    thinking -> "PROCESSING LOCALLY..."
                    modelReady -> "100% OFFLINE AI READY"
                    else -> "LOCAL AI CORE NOT INSTALLED"
                },
                color = Color.Gray,
                fontSize = 10.sp,
                modifier = Modifier.align(Alignment.CenterHorizontally)
            )
            Spacer(Modifier.height(12.dp))

            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(messages) { msg ->
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = if (msg.fromUser) Arrangement.End else Arrangement.Start
                    ) {
                        Surface(
                            color = if (msg.fromUser) Color(0xFF321900) else Color(0xFF141414),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(
                                msg.text,
                                color = if (msg.fromUser) Color(0xFFFFC27A) else Color.LightGray,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(10.dp)
                            )
                        }
                    }
                }
            }

            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    placeholder = { Text("Command JARVIS...", color = Color.Gray) },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFFFF8A00),
                        unfocusedBorderColor = Color.DarkGray,
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White
                    )
                )
                Spacer(Modifier.width(6.dp))
                Button(
                    onClick = onListen,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF8A4700)),
                    contentPadding = PaddingValues(horizontal = 13.dp, vertical = 13.dp)
                ) { Text("MIC") }
                Spacer(Modifier.width(6.dp))
                Button(
                    onClick = { ask(input) },
                    enabled = input.isNotBlank() && !thinking,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFB65F00)),
                    contentPadding = PaddingValues(horizontal = 13.dp, vertical = 13.dp)
                ) { Text("SEND") }
            }
        }
    }
}

private fun buildPrompt(history: List<ChatMessage>): String = buildString {
    append("You are JARVIS, a concise, helpful Android AI assistant. Respond naturally and accurately. Keep spoken answers reasonably short.\n\n")
    history.forEach { message ->
        append(if (message.fromUser) "User: " else "JARVIS: ")
        append(message.text)
        append('\n')
    }
    append("JARVIS:")
}

private class LlmEngine(private val modelFile: File) {
    private val inference: LlmInference
    private val session: LlmInferenceSession

    init {
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)
            .setMaxTokens(512)
            .setPreferredBackend(LlmInference.Backend.CPU)
            .build()
        inference = LlmInference.createFromOptions(AppContextHolder.context, options)
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(40)
            .setTopP(0.95f)
            .setTemperature(0.7f)
            .build()
        session = LlmInferenceSession.createFromOptions(inference, sessionOptions)
    }

    fun ask(prompt: String): String {
        session.addQueryChunk(prompt)
        return session.generateResponse().trim().ifBlank { "I couldn't generate a response." }
    }

    fun close() {
        session.close()
        inference.close()
    }
}

private object AppContextHolder {
    lateinit var context: android.content.Context
}
