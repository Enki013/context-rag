package com.contextrag.quizapp.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoStories
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.data.BookItem
import com.contextrag.quizapp.data.ChapterItem
import com.contextrag.quizapp.viewmodel.UiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BooksScreen(
    state: UiState,
    onSearchChange: (String) -> Unit,
    onSearchSubmit: (String) -> Unit,
    onSelectBook: (BookItem) -> Unit,
    onSelectChapter: (ChapterItem) -> Unit,
    onSetDifficulty: (String) -> Unit,
    onSetCount: (Int) -> Unit,
    onStartQuiz: () -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 16.dp, bottom = 96.dp)
    ) {
        // M3 Arama Kutusu
        item {
            OutlinedTextField(
                value = state.searchQuery,
                onValueChange = onSearchChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Kitap, yazar veya konu ara...") },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = "Ara") },
                trailingIcon = {
                    if (state.searchQuery.isNotEmpty()) {
                        IconButton(onClick = { onSearchChange("") }) {
                            Icon(Icons.Default.Clear, contentDescription = "Temizle")
                        }
                    }
                },
                singleLine = true,
                shape = MaterialTheme.shapes.extraLarge,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.15f)
                )
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Hızlı Arama Önerileri (M3 Filter Chips)
            val quickQueries = listOf("Sherlock Holmes", "Jane Austen", "Frankenstein", "Nutuk", "Felsefe")
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(quickQueries) { q ->
                    FilterChip(
                        selected = state.searchQuery == q,
                        onClick = {
                            onSearchChange(q)
                            onSearchSubmit(q)
                        },
                        label = { Text(q) }
                    )
                }
            }
        }

        // Yükleniyor Göstergesi
        if (state.isSearching) {
            item {
                Box(modifier = Modifier.fillMaxWidth().padding(32.dp), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            }
        }

        // Kitap Arama Sonuçları
        if (state.searchResults.isNotEmpty()) {
            item {
                Text(
                    text = "Arama Sonuçları (${state.searchResults.size})",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            items(state.searchResults) { book ->
                val isSelected = state.selectedBook?.bookId == book.bookId
                ElevatedCard(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelectBook(book) },
                    colors = CardDefaults.elevatedCardColors(
                        containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surface
                    ),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = if (isSelected) 4.dp else 1.dp),
                    shape = MaterialTheme.shapes.medium
                ) {
                    ListItem(
                        headlineContent = {
                            Text(
                                text = book.title,
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        },
                        supportingContent = {
                            Column {
                                Text(
                                    text = book.author ?: "Bilinmiyor",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                if (!book.subjects.isNullOrBlank()) {
                                    Text(
                                        text = book.subjects,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.outline,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }
                        },
                        leadingContent = {
                            FilledTonalIconButton(onClick = { onSelectBook(book) }) {
                                Icon(Icons.Default.MenuBook, contentDescription = null)
                            }
                        },
                        trailingContent = {
                            if (isSelected) {
                                Icon(
                                    Icons.Default.Check,
                                    contentDescription = "Seçildi",
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    )
                }
            }
        }

        // Seçili Kitap Bölüm Listesi & Quiz Parametreleri
        state.selectedBook?.let { selected ->
            item {
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                Text(
                    text = "📖 ${selected.title} - Bölümler",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            if (state.isLoadingChapters) {
                item {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            } else if (state.chaptersData != null) {
                items(state.chaptersData.chapters) { chapter ->
                    val isChSelected = state.selectedChapter?.chapterIndex == chapter.chapterIndex
                    OutlinedCard(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onSelectChapter(chapter) },
                        colors = CardDefaults.outlinedCardColors(
                            containerColor = if (isChSelected) MaterialTheme.colorScheme.secondaryContainer
                            else MaterialTheme.colorScheme.surface
                        ),
                        shape = MaterialTheme.shapes.medium
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = chapter.title,
                                    style = MaterialTheme.typography.titleSmall,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f)
                                )
                                AssistChip(
                                    onClick = { },
                                    label = { Text("Havuz: ${chapter.poolCount}") },
                                    leadingIcon = { Icon(Icons.Default.Bolt, contentDescription = null, modifier = Modifier.size(16.dp)) }
                                )
                            }

                            Spacer(modifier = Modifier.height(8.dp))

                            // İlerleme Çubuğu (%0 - %100)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                LinearProgressIndicator(
                                    progress = (chapter.progressPercent / 100f).coerceIn(0f, 1f),
                                    modifier = Modifier.weight(1f)
                                )
                                Text(
                                    text = "%${chapter.progressPercent.toInt()}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.outline
                                )
                            }
                        }
                    }
                }

                // Zorluk ve Soru Sayısı Ayarları
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)),
                        shape = MaterialTheme.shapes.large
                    ) {
                        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            Text(
                                text = "Quiz Ayarları",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )

                            // Zorluk Seçimi
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Zorluk:", style = MaterialTheme.typography.bodyMedium)
                                listOf("kolay", "orta", "zor").forEach { diff ->
                                    FilterChip(
                                        selected = state.quizDifficulty == diff,
                                        onClick = { onSetDifficulty(diff) },
                                        label = { Text(diff.replaceFirstChar { it.uppercase() }) }
                                    )
                                }
                            }

                            // Soru Sayısı
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Soru Sayısı:", style = MaterialTheme.typography.bodyMedium)
                                listOf(3, 5, 10).forEach { cnt ->
                                    FilterChip(
                                        selected = state.quizCount == cnt,
                                        onClick = { onSetCount(cnt) },
                                        label = { Text("$cnt Soru") }
                                    )
                                }
                            }

                            // Başlat Butonu
                            Button(
                                onClick = onStartQuiz,
                                modifier = Modifier.fillMaxWidth(),
                                enabled = state.selectedChapter != null && !state.isLoadingQuiz,
                                shape = CircleShape
                            ) {
                                if (state.isLoadingQuiz) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(20.dp),
                                        color = MaterialTheme.colorScheme.onPrimary,
                                        strokeWidth = 2.dp
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Soru Bankasından Çekiliyor...")
                                } else {
                                    Icon(Icons.Default.PlayArrow, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Bölüm Quizini Başlat")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
