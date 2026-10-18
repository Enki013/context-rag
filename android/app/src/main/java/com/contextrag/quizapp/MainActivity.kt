package com.contextrag.quizapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.contextrag.quizapp.ui.MainScreen
import com.contextrag.quizapp.ui.theme.ContextRagTheme
import com.contextrag.quizapp.viewmodel.QuizViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: QuizViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            ContextRagTheme {
                MainScreen(viewModel = viewModel)
            }
        }
    }
}
