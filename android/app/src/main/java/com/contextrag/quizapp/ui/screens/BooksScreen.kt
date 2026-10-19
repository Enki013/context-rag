package com.contextrag.quizapp.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoStories
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.contextrag.quizapp.data.BookItem
import com.contextrag.quizapp.ui.theme.CacheHitGreen
import com.contextrag.quizapp.ui.theme.CacheHitGreenContainer
import com.contextrag.quizapp.ui.theme.OnCacheHitGreen
import com.contextrag.quizapp.viewmodel.UiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BooksScreen(
    state: UiState,
    onSearchChange: (String) -> Unit,
    onSearchSubmit: (String) -> Unit,
    onSelectBook: (BookItem) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(top = 16.dp, bottom = 96.dp)
    ) {
        // M3 Arama Kutusu & Hızlı Filtreler
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

            Spacer(modifier = Modifier.height(10.dp))

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
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(48.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        CircularProgressIndicator()
                        Text(
                            text = "Gutenberg ve yerel kütüphane taranıyor...",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }

        // Kitap Arama Sonuçları
        if (state.searchResults.isNotEmpty()) {
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Bulunan Kitaplar",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Badge(
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                        contentColor = MaterialTheme.colorScheme.onPrimaryContainer
                    ) {
                        Text(
                            text = "${state.searchResults.size} Kitap",
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }

            items(state.searchResults) { book ->
                ElevatedCard(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelectBook(book) },
                    colors = CardDefaults.elevatedCardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 1.dp),
                    shape = MaterialTheme.shapes.large
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
                            Column(
                                modifier = Modifier.padding(top = 4.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "${book.author ?: "Bilinmeyen Yazar"} • ID: #${book.bookId}",
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
                                if (book.isCachedLocally) {
                                    SuggestionChip(
                                        onClick = { onSelectBook(book) },
                                        label = { Text("⚡ Cihazda Hazır (0ms)") },
                                        colors = SuggestionChipDefaults.suggestionChipColors(
                                            containerColor = CacheHitGreenContainer,
                                            labelColor = OnCacheHitGreen
                                        ),
                                        border = null,
                                        modifier = Modifier.padding(top = 2.dp)
                                    )
                                }
                            }
                        },
                        leadingContent = {
                            FilledTonalIconButton(
                                onClick = { onSelectBook(book) },
                                colors = IconButtonDefaults.filledTonalIconButtonColors(
                                    containerColor = if (book.isCachedLocally) CacheHitGreenContainer
                                    else MaterialTheme.colorScheme.secondaryContainer
                                )
                            ) {
                                Icon(
                                    imageVector = if (book.isCachedLocally) Icons.Default.AutoStories else Icons.Default.MenuBook,
                                    contentDescription = null,
                                    tint = if (book.isCachedLocally) OnCacheHitGreen else MaterialTheme.colorScheme.onSecondaryContainer
                                )
                            }
                        },
                        trailingContent = {
                            IconButton(onClick = { onSelectBook(book) }) {
                                Icon(
                                    Icons.Default.ChevronRight,
                                    contentDescription = "Bölümleri Gör",
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    )
                }
            }
        } else if (!state.isSearching) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 64.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            Icons.Default.MenuBook,
                            contentDescription = null,
                            modifier = Modifier.size(64.dp),
                            tint = MaterialTheme.colorScheme.outlineVariant
                        )
                        Text(
                            text = "Aramak için bir anahtar kelime girin",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}
