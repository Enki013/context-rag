package com.contextrag.quizapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Dns
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.viewmodel.UiState

@Composable
fun SettingsScreen(
    state: UiState,
    onUpdateBaseUrl: (String) -> Unit,
    onRefreshHealth: () -> Unit
) {
    var urlInput by remember(state.baseUrl) { mutableStateOf(state.baseUrl) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Sunucu Bağlantı Ayarı (M3 ElevatedCard)
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Icons.Default.Dns, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Text("API Sunucu Yapılandırması", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                }

                Text(
                    text = "Android Emülatör için varsayılan 'http://10.0.2.2:8000' kullanılır. Fiziksel cihaz için yerel Wi-Fi IP'nizi girin (örn: 'http://192.168.1.100:8000').",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = urlInput,
                    onValueChange = { urlInput = it },
                    label = { Text("Base URL") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.medium
                )

                Button(
                    onClick = {
                        onUpdateBaseUrl(urlInput)
                        onRefreshHealth()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.full
                ) {
                    if (state.isCheckingHealth) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Bağlantı Kontrol Ediliyor...")
                    } else {
                        Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Bağlan ve Test Et")
                    }
                }
            }
        }

        // Sunucu Durum Bilgisi (Health)
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Backend & Model Sağlığı", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)

                val health = state.health
                if (health != null) {
                    ListItem(
                        headlineContent = { Text("Sunucu Durumu") },
                        supportingContent = { Text("FastAPI Çevrimiçi (Port 8000)") },
                        leadingContent = {
                            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = CacheHitGreen)
                        },
                        trailingContent = {
                            SuggestionChip(
                                onClick = {},
                                label = { Text("Online") },
                                colors = SuggestionChipDefaults.suggestionChipColors(
                                    containerColor = CacheHitGreenContainer,
                                    labelColor = OnCacheHitGreen
                                )
                            )
                        }
                    )

                    HorizontalDivider()

                    ListItem(
                        headlineContent = { Text("Ollama LLM") },
                        supportingContent = { Text(health.llmModel) },
                        leadingContent = {
                            Icon(Icons.Default.Bolt, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        }
                    )

                    ListItem(
                        headlineContent = { Text("Embedding Modeli") },
                        supportingContent = { Text(health.embeddingModel) },
                        leadingContent = {
                            Icon(Icons.Default.Speed, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
                        }
                    )

                    ListItem(
                        headlineContent = { Text("Kitap Kataloğu") },
                        supportingContent = { Text("${health.totalBooks} Kitap İndeksli (SQLite FTS5)") },
                        leadingContent = {
                            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.tertiary)
                        }
                    )
                } else {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.padding(vertical = 8.dp)
                    ) {
                        Icon(Icons.Default.Error, contentDescription = null, tint = MaterialTheme.colorScheme.error)
                        Text(
                            text = "Sunucuya ulaşılamıyor. Lütfen FastAPI'nin çalıştığından emin olun.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }

        // Mimari Özet Kartı
        OutlinedCard(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 80.dp),
            shape = MaterialTheme.shapes.large
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Default.Security, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Text("Headless Engine Mimarisi", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                }
                Text("• 0ms Cache Hit: Bir kez üretilen sorular kalıcı havuzda saklanır ve anında servis edilir.", style = MaterialTheme.typography.bodySmall)
                Text("• Hile Koruması: Mobil istemciye giden sorularda doğru şıklar sızdırılmaz (null döner).", style = MaterialTheme.typography.bodySmall)
                Text("• Anti-Spoiler: Quizler okuma ilerlemesine (bölüm sınırlarına) kilitlidir.", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
