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
import com.contextrag.quizapp.ui.components.expressiveBounceClick
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.M3Shapes
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.ui.theme.titleMediumEmphasized
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
        // 1. Sunucu Bağlantı Ayarı (M3 Card with Surface Container)
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = M3Shapes.CornerLarge,
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainerLow
            )
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Icons.Default.Dns, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Text("API Sunucu Yapılandırması", style = MaterialTheme.typography.titleMediumEmphasized)
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
                    shape = M3Shapes.CornerMedium,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.4f),
                        unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.2f)
                    )
                )

                Button(
                    onClick = {
                        onUpdateBaseUrl(urlInput)
                        onRefreshHealth()
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                        .expressiveBounceClick {
                            onUpdateBaseUrl(urlInput)
                            onRefreshHealth()
                        },
                    shape = M3Shapes.CornerFull
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
                        Text("Bağlan ve Test Et", fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        // 2. Sunucu Durum Bilgisi (Health)
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = M3Shapes.CornerLarge,
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainerLow
            )
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Backend & Model Sağlığı", style = MaterialTheme.typography.titleMediumEmphasized)

                val health = state.health
                if (health != null) {
                    ListItem(
                        headlineContent = { Text("Sunucu Durumu", fontWeight = FontWeight.Medium) },
                        supportingContent = { Text("FastAPI Çevrimiçi (Port 8000)") },
                        leadingContent = {
                            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = CacheHitGreen)
                        },
                        trailingContent = {
                            Surface(
                                shape = M3Shapes.CornerFull,
                                color = CacheHitGreenContainer
                            ) {
                                Text(
                                    text = "Online",
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = OnCacheHitGreen,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        },
                        colors = ListItemDefaults.colors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)
                    )

                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

                    ListItem(
                        headlineContent = { Text("Ollama LLM", fontWeight = FontWeight.Medium) },
                        supportingContent = { Text(health.llmModel) },
                        leadingContent = {
                            Icon(Icons.Default.Bolt, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        },
                        colors = ListItemDefaults.colors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)
                    )

                    ListItem(
                        headlineContent = { Text("Embedding Modeli", fontWeight = FontWeight.Medium) },
                        supportingContent = { Text(health.embeddingModel) },
                        leadingContent = {
                            Icon(Icons.Default.Speed, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
                        },
                        colors = ListItemDefaults.colors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)
                    )

                    ListItem(
                        headlineContent = { Text("Kitap Kataloğu", fontWeight = FontWeight.Medium) },
                        supportingContent = { Text("${health.totalBooks} Kitap İndeksli (SQLite FTS5)") },
                        leadingContent = {
                            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.tertiary)
                        },
                        colors = ListItemDefaults.colors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)
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

        // 3. Mimari Özet Kartı
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 80.dp),
            shape = M3Shapes.CornerLarge,
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            )
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Default.Security, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Text("Headless Engine Mimarisi", style = MaterialTheme.typography.titleMediumEmphasized)
                }
                Text("• 0ms Cache Hit: Bir kez üretilen sorular kalıcı havuzda saklanır ve anında servis edilir.", style = MaterialTheme.typography.bodySmall)
                Text("• Hile Koruması: Mobil istemciye giden sorularda doğru şıklar sızdırılmaz (null döner).", style = MaterialTheme.typography.bodySmall)
                Text("• Anti-Spoiler: Quizler okuma ilerlemesine (bölüm sınırlarına) kilitlidir.", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
