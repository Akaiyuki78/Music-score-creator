(() => {
  const editor = document.getElementById("chordpro-editor");
  const selCapo = document.getElementById("sel-capo");
  const chkCapoShapes = document.getElementById("chk-capo-shapes");
  const chkNashville = document.getElementById("chk-nashville");
  const selPinyinStyle = document.getElementById("sel-pinyin-style");
  const chkDiagrams = document.getElementById("chk-diagrams");
  const lblTranspose = document.getElementById("lbl-transpose");

  const previewTitle = document.getElementById("preview-title");
  const previewSubtitle = document.getElementById("preview-subtitle");
  const previewArtist = document.getElementById("preview-artist");
  const previewMetaBar = document.getElementById("preview-meta-bar");
  const previewDiagrams = document.getElementById("preview-diagrams");
  const previewSections = document.getElementById("preview-sections");

  const boxConcertKey = document.getElementById("box-concert-key");
  const boxOutputKey = document.getElementById("box-output-key");

  const uploadDialog = document.getElementById("upload-dialog");
  const uploadStatus = document.getElementById("upload-status");

  let transposeSemitones = 0;
  let lastRecommendedCapo = 0;
  let debounceTimer = null;
  let currentPreviewData = null;

  // Pitch mapping for client-side inverse transposition
  const SEMITONES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const FLAT_SEMITONES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
  const NOTE_MAP = {
    C: 0, "B#": 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3, E: 4, Fb: 4, "E#": 5,
    F: 5, "F#": 6, Gb: 6, G: 7, "G#": 8, Ab: 8, A: 9, "A#": 10, Bb: 10, B: 11, Cb: 11
  };

  function transposeChordJS(chordStr, semitones, preferFlats = false) {
    if (!chordStr || !semitones) return chordStr;
    const match = chordStr.match(/^([A-Ga-g][#b]?)(.*)$/);
    if (!match) return chordStr;
    let root = match[1];
    root = root.charAt(0).toUpperCase() + root.slice(1);
    const suffix = match[2] || "";
    if (!(root in NOTE_MAP)) return chordStr;
    const oldPitch = NOTE_MAP[root];
    const newPitch = ((oldPitch + semitones) % 12 + 12) % 12;
    const table = preferFlats ? FLAT_SEMITONES : SEMITONES;
    return `${table[newPitch]}${suffix}`;
  }

  function getEffectiveSemitoneShift() {
    const capoFret = chkCapoShapes.checked ? parseInt(selCapo.value, 10) || 0 : 0;
    return transposeSemitones - capoFret;
  }

  function updateSingleChord(secIdx, lineIdx, tokIdx, newChord) {
    if (!currentPreviewData) return;
    const sec = currentPreviewData.sections[secIdx];
    if (!sec) return;
    const line = sec.lines[lineIdx];
    if (!line) return;
    const tok = line.tokens[tokIdx];
    if (!tok) return;

    if (!newChord) {
      tok.chord = null;
      tok.raw_chord = null;
    } else {
      // Invert any active transposition or Capo shift so the master document
      // stores the true concert chord, and the screen displays EXACTLY what the user typed!
      const effectiveShift = getEffectiveSemitoneShift();
      const rawConcertChord = transposeChordJS(newChord, -effectiveShift, true);
      tok.chord = newChord;
      tok.raw_chord = rawConcertChord;
    }

    reconstructChordProFromPreview();
  }

  function moveChordInChordPro(srcSec, srcLine, srcTok, dstSec, dstLine, dstTok, chordValue) {
    if (!currentPreviewData) return;
    try {
      const srcL = currentPreviewData.sections[srcSec].lines[srcLine];
      const dstL = currentPreviewData.sections[dstSec].lines[dstLine];
      let rawChord = chordValue;
      if (srcL && srcL.tokens[srcTok]) {
        rawChord = srcL.tokens[srcTok].raw_chord || chordValue;
        srcL.tokens[srcTok].chord = null;
        srcL.tokens[srcTok].raw_chord = null;
      }
      if (dstL && dstL.tokens[dstTok]) {
        dstL.tokens[dstTok].chord = chordValue;
        dstL.tokens[dstTok].raw_chord = rawChord;
      }
      reconstructChordProFromPreview();
    } catch (e) {
      console.error("Error moving chord:", e);
    }
  }

  function reconstructChordProFromPreview() {
    if (!currentPreviewData) return;
    const lines = [];
    lines.push(`{title: ${currentPreviewData.title || "Untitled"}}`);
    if (currentPreviewData.subtitle) lines.push(`{subtitle: ${currentPreviewData.subtitle}}`);
    if (currentPreviewData.artist) lines.push(`{artist: ${currentPreviewData.artist}}`);
    lines.push(`{key: ${currentPreviewData.concert_key || "C"}}`);
    if (currentPreviewData.capo > 0) lines.push(`{capo: ${currentPreviewData.capo}}`);
    lines.push(`{tempo: ${currentPreviewData.tempo_bpm || 120}}`);
    lines.push(`{time: ${currentPreviewData.time_signature || "4/4"}}`);
    lines.push("");

    currentPreviewData.sections.forEach((sec) => {
      const sType = (sec.section_type || "verse").toLowerCase();
      if (sType === "chorus") lines.push(`{start_of_chorus: ${sec.name}}`);
      else if (sType === "bridge") lines.push(`{start_of_bridge: ${sec.name}}`);
      else lines.push(`{start_of_verse: ${sec.name}}`);

      sec.lines.forEach((ln) => {
        if (ln.is_comment) {
          lines.push(`{comment: ${ln.comment_text}}`);
          return;
        }
        let rowStr = "";
        (ln.tokens || []).forEach((t) => {
          const chordToWrite = t.raw_chord || t.chord;
          if (chordToWrite) {
            rowStr += `[${chordToWrite}]`;
          }
          rowStr += t.text || "";
        });
        lines.push(rowStr);
      });

      if (sType === "chorus") lines.push("{end_of_chorus}");
      else if (sType === "bridge") lines.push("{end_of_bridge}");
      else lines.push("{end_of_verse}");
      lines.push("");
    });

    editor.value = lines.join("\n").trim() + "\n";
    refreshPreview();
  }

  function getSettings() {
    const pyVal = selPinyinStyle.value;
    return {
      chordpro_text: editor.value,
      transpose_semitones: transposeSemitones,
      capo: parseInt(selCapo.value, 10) || 0,
      use_capo_shapes: chkCapoShapes.checked,
      use_nashville: chkNashville.checked,
      tone_style: pyVal === "hidden" ? "marks" : pyVal,
      show_pinyin: pyVal !== "hidden",
    };
  }

  function renderGuitarSvg(frets) {
    const active = frets.filter((f) => f > 0);
    const minFret = active.length ? Math.min(...active) : 1;
    const maxFret = active.length ? Math.max(...active) : 1;
    const baseFret = maxFret <= 4 ? 1 : minFret;

    const w = 68;
    const h = 76;
    const gx = 14;
    const gy = 16;
    const gw = 44;
    const gh = 50;
    const dx = gw / 5;
    const dy = gh / 4;

    let svg = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;
    // Nut or base fret label
    if (baseFret === 1) {
      svg += `<line x1="${gx}" y1="${gy}" x2="${gx + gw}" y2="${gy}" stroke="currentColor" stroke-width="3" />`;
    } else {
      svg += `<line x1="${gx}" y1="${gy}" x2="${gx + gw}" y2="${gy}" stroke="currentColor" stroke-width="1" />`;
      svg += `<text x="${gx - 3}" y="${gy + dy * 0.75}" font-size="8" text-anchor="end" fill="currentColor">${baseFret}fr</text>`;
    }

    for (let f = 1; f <= 4; f++) {
      const y = gy + f * dy;
      svg += `<line x1="${gx}" y1="${y}" x2="${gx + gw}" y2="${y}" stroke="currentColor" stroke-opacity="0.45" stroke-width="1" />`;
    }
    for (let s = 0; s < 6; s++) {
      const x = gx + s * dx;
      svg += `<line x1="${x}" y1="${gy}" x2="${x}" y2="${gy + gh}" stroke="currentColor" stroke-opacity="0.6" stroke-width="1" />`;
    }

    frets.forEach((fVal, sIdx) => {
      const x = gx + sIdx * dx;
      if (fVal === -1) {
        svg += `<text x="${x}" y="${gy - 4}" font-size="9" text-anchor="middle" fill="#64748b">×</text>`;
      } else if (fVal === 0) {
        svg += `<circle cx="${x}" cy="${gy - 6}" r="2.6" stroke="currentColor" stroke-width="1.1" fill="none" />`;
      } else {
        const rel = fVal - baseFret + 1;
        if (rel >= 1 && rel <= 4) {
          const cy = gy + (rel - 0.5) * dy;
          svg += `<circle cx="${x}" cy="${cy}" r="3.6" fill="#2563eb" />`;
        }
      }
    });

    svg += `</svg>`;
    return svg;
  }

  function updatePreviewDOM(preview) {
    currentPreviewData = preview;
    const settings = getSettings();
    lastRecommendedCapo = preview.recommended_capo || 0;

    previewTitle.textContent = preview.title || "Untitled Song";
    previewSubtitle.textContent = preview.subtitle || "";
    previewArtist.textContent = preview.artist || "";

    // Update the single unified Key & Pitch Box
    if (boxConcertKey) {
      boxConcertKey.textContent = preview.concert_key || "C";
    }
    if (boxOutputKey) {
      boxOutputKey.textContent = preview.play_key || preview.concert_key || "C";
    }

    const metaParts = [];
    if (preview.capo > 0 && settings.use_capo_shapes) {
      metaParts.push(`<span>Capo: <strong>Fret ${preview.capo}</strong> (Play in <strong>${preview.play_key}</strong>)</span>`);
    } else {
      metaParts.push(`<span>Key: <strong>${preview.play_key}</strong></span>`);
    }
    if (preview.recommended_capo > 0 && preview.capo !== preview.recommended_capo) {
      metaParts.push(`<span>💡 Suggested: <strong>Capo ${preview.recommended_capo}</strong> for easy open chords</span>`);
    }
    metaParts.push(`<span>Tempo: <strong>${preview.tempo_bpm} BPM</strong></span>`);
    metaParts.push(`<span>Time: <strong>${preview.time_signature}</strong></span>`);
    previewMetaBar.innerHTML = metaParts.join("");

    // Render Guitar Chord Diagrams
    if (chkDiagrams.checked && !settings.use_nashville && preview.chord_diagrams && preview.chord_diagrams.length) {
      previewDiagrams.style.display = "flex";
      previewDiagrams.innerHTML = preview.chord_diagrams
        .map(
          (cd) => `
          <div class="chord-card">
            <div class="chord-card-title">${cd.symbol}</div>
            ${renderGuitarSvg(cd.frets)}
          </div>`
        )
        .join("");
    } else {
      previewDiagrams.style.display = "none";
      previewDiagrams.innerHTML = "";
    }

    // Render Sections & 3-tier lines
    previewSections.innerHTML = "";
    (preview.sections || []).forEach((sec) => {
      const secEl = document.createElement("div");
      const isChorus =
        (sec.section_type || "").toLowerCase() === "chorus" ||
        sec.name.toLowerCase().includes("chorus") ||
        sec.name.includes("副歌");
      secEl.className = `score-section${isChorus ? " is-chorus" : ""}`;

      const titleEl = document.createElement("div");
      titleEl.className = "section-title";
      titleEl.textContent = `[${sec.name}]`;
      secEl.appendChild(titleEl);

      (sec.lines || []).forEach((ln) => {
        if (ln.is_comment) {
          const cEl = document.createElement("div");
          cEl.className = "score-comment";
          cEl.textContent = ln.comment_text;
          secEl.appendChild(cEl);
          return;
        }
        const lineEl = document.createElement("div");
        lineEl.className = "score-line";

        const hasLineChord = (ln.tokens || []).some((t) => t.chord);
        const hasLinePinyin = settings.show_pinyin && (ln.tokens || []).some((t) => t.pinyin);

        (ln.tokens || []).forEach((tok, tokIdx) => {
          const cell = document.createElement("div");
          const isSpace = tok.text === " " && !tok.chord;
          cell.className = `token-cell${tok.is_cjk ? " is-cjk" : ""}${isSpace ? " is-space" : ""}`;
          cell.dataset.secIdx = String(preview.sections.indexOf(sec));
          cell.dataset.lineIdx = String(sec.lines.indexOf(ln));
          cell.dataset.tokIdx = String(tokIdx);

          // Drag and Drop listeners on target cells
          cell.addEventListener("dragover", (e) => {
            e.preventDefault();
            cell.classList.add("drag-over");
          });
          cell.addEventListener("dragleave", () => {
            cell.classList.remove("drag-over");
          });
          cell.addEventListener("drop", (e) => {
            e.preventDefault();
            cell.classList.remove("drag-over");
            const draggedChord = e.dataTransfer.getData("text/plain");
            const srcSec = parseInt(e.dataTransfer.getData("application/x-sec"), 10);
            const srcLine = parseInt(e.dataTransfer.getData("application/x-line"), 10);
            const srcTok = parseInt(e.dataTransfer.getData("application/x-tok"), 10);

            if (draggedChord) {
              moveChordInChordPro(srcSec, srcLine, srcTok, parseInt(cell.dataset.secIdx, 10), parseInt(cell.dataset.lineIdx, 10), tokIdx, draggedChord);
            }
          });

          const cRow = document.createElement("div");
          cRow.className = "tier-chord";

          if (tok.chord) {
            const chip = document.createElement("span");
            chip.className = "chord-chip";
            chip.innerHTML = `
              <span class="chord-chip-text">${tok.chord}</span>
              <span class="chord-chip-del" title="Delete this chord">✕</span>
            `;
            chip.title = "Click chord name to edit, click ✕ to delete, or drag to another word/space";
            chip.draggable = true;

            chip.addEventListener("dragstart", (e) => {
              e.dataTransfer.setData("text/plain", tok.raw_chord || tok.chord);
              e.dataTransfer.setData("application/x-sec", cell.dataset.secIdx);
              e.dataTransfer.setData("application/x-line", cell.dataset.lineIdx);
              e.dataTransfer.setData("application/x-tok", cell.dataset.tokIdx);
            });

            chip.addEventListener("click", (e) => {
              e.stopPropagation();
              // If user clicked the little '✕' delete button
              if (e.target.classList.contains("chord-chip-del")) {
                updateSingleChord(parseInt(cell.dataset.secIdx, 10), parseInt(cell.dataset.lineIdx, 10), tokIdx, null);
                return;
              }
              // Otherwise edit the chord name
              const newChord = prompt(`Edit chord above "${tok.text.trim() || 'space'}" (leave empty to remove):`, tok.chord || tok.raw_chord);
              if (newChord !== null) {
                updateSingleChord(parseInt(cell.dataset.secIdx, 10), parseInt(cell.dataset.lineIdx, 10), tokIdx, newChord.trim());
              }
            });
            cRow.appendChild(chip);
          } else {
            // Empty placeholder that can be clicked to add a chord (works on words, syllables, AND spaces!)
            const ph = document.createElement("span");
            ph.className = isSpace ? "chord-placeholder is-space-ph" : "chord-placeholder";
            ph.title = isSpace ? "Click to add a passing chord in this space" : `Click to add chord above "${tok.text}"`;
            ph.addEventListener("click", (e) => {
              e.stopPropagation();
              const label = isSpace ? "space / passing beat" : `"${tok.text}"`;
              const addChord = prompt(`Add chord above ${label}:`, "G");
              if (addChord && addChord.trim()) {
                updateSingleChord(parseInt(cell.dataset.secIdx, 10), parseInt(cell.dataset.lineIdx, 10), tokIdx, addChord.trim());
              }
            });
            cRow.appendChild(ph);
          }
          cell.appendChild(cRow);

          if (hasLinePinyin) {
            const pRow = document.createElement("div");
            pRow.className = "tier-pinyin";
            pRow.textContent = tok.pinyin || "";
            cell.appendChild(pRow);
          }

          const lRow = document.createElement("div");
          lRow.className = "tier-lyric";
          lRow.textContent = tok.text || "";
          cell.appendChild(lRow);

          lineEl.appendChild(cell);
        });

        secEl.appendChild(lineEl);
      });

      previewSections.appendChild(secEl);
    });
  }

  async function refreshPreview() {
    const req = getSettings();
    const res = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (res.ok) {
      const data = await res.json();
      updatePreviewDOM(data.preview);
    }
  }

  function scheduleRefresh() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(refreshPreview, 180);
  }

  async function loadDemo(preset = "ge_qian") {
    const res = await fetch(`/api/demo?preset=${encodeURIComponent(preset)}&tone_style=${encodeURIComponent(getSettings().tone_style)}`);
    if (res.ok) {
      const data = await res.json();
      editor.value = data.chordpro_text;
      transposeSemitones = 0;
      lblTranspose.textContent = "0 st";
      selCapo.value = String(data.preview.capo || 0);
      updatePreviewDOM(data.preview);
    }
  }

  async function downloadFile(endpoint, fallbackName) {
    const req = getSettings();
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fallbackName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Event Listeners
  editor.addEventListener("input", scheduleRefresh);
  selCapo.addEventListener("change", refreshPreview);
  chkCapoShapes.addEventListener("change", refreshPreview);
  chkNashville.addEventListener("change", refreshPreview);
  selPinyinStyle.addEventListener("change", refreshPreview);
  chkDiagrams.addEventListener("change", refreshPreview);

  document.getElementById("btn-trans-down").addEventListener("click", () => {
    transposeSemitones -= 1;
    lblTranspose.textContent = `${transposeSemitones > 0 ? "+" : ""}${transposeSemitones} st`;
    refreshPreview();
  });

  document.getElementById("btn-trans-up").addEventListener("click", () => {
    transposeSemitones += 1;
    lblTranspose.textContent = `${transposeSemitones > 0 ? "+" : ""}${transposeSemitones} st`;
    refreshPreview();
  });

  document.getElementById("btn-auto-capo").addEventListener("click", () => {
    selCapo.value = String(lastRecommendedCapo);
    chkCapoShapes.checked = true;
    refreshPreview();
  });

  document.getElementById("btn-load-geqian")?.addEventListener("click", () => loadDemo("ge_qian"));
  document.getElementById("btn-load-demo")?.addEventListener("click", () => loadDemo("peng_you"));
  document.getElementById("btn-load-shout")?.addEventListener("click", () => loadDemo("shout"));

  document.getElementById("btn-quick-import")?.addEventListener("click", async () => {
    const qInput = document.getElementById("quick-youtube-input");
    const val = (qInput?.value || "").trim();
    if (!val) {
      alert("Please enter a song title or paste a YouTube URL!");
      qInput?.focus();
      return;
    }
    if (val.includes("youtube.com") || val.includes("youtu.be")) {
      scrapeYouTube(val);
    } else {
      searchAndDisplayChords(val);
    }
  });

  document.getElementById("quick-youtube-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      document.getElementById("btn-quick-import")?.click();
    }
  });

  async function scrapeYouTube(ytUrl) {
    const versionBox = document.getElementById("version-results-box");
    const pillList = document.getElementById("version-pill-list");
    const statusHint = document.getElementById("version-results-hint");
    if (versionBox) versionBox.classList.remove("hidden");
    if (statusHint) statusHint.textContent = "Scraping YouTube video title & lyrics...";
    if (pillList) pillList.innerHTML = `<span style="font-size:0.85rem; color:var(--text-muted); padding:4px;">Scraping video info...</span>`;

    try {
      const res = await fetch(`/api/scrape-youtube?url=${encodeURIComponent(ytUrl)}`);
      if (!res.ok) throw new Error("Could not scrape YouTube video");
      const data = await res.json();

      // If chord versions found, render pills and load top version
      if (data.chord_versions && data.chord_versions.length > 0) {
        statusHint.textContent = `Scraped "${data.title}". Found ${data.chord_versions.length} verified chord versions:`;
        renderVersionPills(data.chord_versions);
        loadCommunityTab(data.chord_versions[0].tab_url);
      } else if (data.lyrics) {
        // No chords found, but lyrics scraped: populate editor ready for user chords
        editor.value = `{title: ${data.title}}\n{artist: ${data.artist}}\n{time: 4/4}\n\n${data.lyrics}`;
        refreshPreview();
        statusHint.textContent = `Scraped lyrics for "${data.title}". Add chords using [Chord] format.`;
        pillList.innerHTML = `<span style="font-size:0.85rem; color:var(--text-muted); padding:4px;">Lyrics loaded. Type chords like [G], [D] directly above lyrics!</span>`;
      } else {
        statusHint.textContent = `Scraped "${data.title}". No lyrics or community chords found.`;
        pillList.innerHTML = "";
      }
    } catch (e) {
      if (statusHint) statusHint.textContent = `Error: ${e.message}`;
    }
  }

  async function searchAndDisplayChords(query) {
    const versionBox = document.getElementById("version-results-box");
    const pillList = document.getElementById("version-pill-list");
    const statusHint = document.getElementById("version-results-hint");
    if (!versionBox || !pillList) return;

    versionBox.classList.remove("hidden");
    statusHint.textContent = "Searching verified community chords...";
    pillList.innerHTML = `<span style="font-size: 0.85rem; color: var(--text-muted); padding: 4px;">Searching across 1M+ guitar chord sheets...</span>`;

    try {
      const res = await fetch(`/api/chords/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      const results = data.results || [];
      if (results.length === 0) {
        pillList.innerHTML = `<span style="font-size: 0.85rem; color: var(--text-muted); padding: 4px;">No community chord sheets found for "${query}". Try searching with artist name.</span>`;
        statusHint.textContent = "";
        return;
      }

      statusHint.textContent = `Found ${results.length} verified versions. Auto-loaded recommended version:`;
      renderVersionPills(results);

      // Automatically load the highest-rated version immediately!
      loadCommunityTab(results[0].tab_url);
    } catch (e) {
      statusHint.textContent = `Error: ${e.message}`;
    }
  }

  function renderVersionPills(results) {
    const pillList = document.getElementById("version-pill-list");
    if (!pillList) return;
    pillList.innerHTML = "";

    results.forEach((item, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `version-pill ${idx === 0 ? "active" : ""}`;
      btn.innerHTML = `
        <strong>${item.version_desc || `Ver ${item.version}`}</strong>
        <span>${item.artist_name || item.song_name}</span>
        <span class="pill-tag">⭐ ${item.rating} (${item.votes})</span>
      `;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".version-pill").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        loadCommunityTab(item.tab_url);
      });
      pillList.appendChild(btn);
    });
  }

  async function loadCommunityTab(tabUrl) {
    try {
      const res = await fetch("/api/chords/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tab_url: tabUrl, tone_style: getSettings().tone_style }),
      });
      if (res.ok) {
        const data = await res.json();
        editor.value = data.chordpro_text;
        selCapo.value = String(data.capo || 0);
        transposeSemitones = 0;
        lblTranspose.textContent = "0 st";
        updatePreviewDOM(data.preview);
      }
    } catch (e) {
      console.error("Failed to load community tab:", e);
    }
  }

  // Worship Setlist Sequence State
  let setlistSongs = [];
  let activeSetlistIndex = -1;

  function loadSetlistFromStorage() {
    try {
      const saved = localStorage.getItem("worship_setlist");
      if (saved) {
        setlistSongs = JSON.parse(saved);
      }
    } catch (e) {
      setlistSongs = [];
    }
    renderSetlistTray();
  }

  function saveSetlistToStorage() {
    try {
      localStorage.setItem("worship_setlist", JSON.stringify(setlistSongs));
    } catch (e) {}
    renderSetlistTray();
  }

  function addCurrentSongToSetlist() {
    const title = previewTitle.textContent.trim() || "Untitled Song";
    const outKey = (boxOutputKey ? boxOutputKey.textContent.trim() : "") || "C";
    const songEntry = {
      id: Date.now(),
      title: title,
      artist: previewArtist.textContent.trim() || "",
      key: outKey,
      chordpro_text: editor.value,
      transpose_semitones: transposeSemitones,
      capo: parseInt(selCapo.value, 10) || 0,
      use_capo_shapes: chkCapoShapes.checked,
      use_nashville: chkNashville.checked,
      tone_style: selPinyinStyle.value,
      show_pinyin: selPinyinStyle.value !== "hidden",
    };

    setlistSongs.push(songEntry);
    activeSetlistIndex = setlistSongs.length - 1;
    saveSetlistToStorage();
  }

  function renderSetlistTray() {
    const listEl = document.getElementById("setlist-items");
    const countEl = document.getElementById("setlist-count-badge");
    if (!listEl) return;

    if (countEl) {
      countEl.textContent = `${setlistSongs.length} song${setlistSongs.length === 1 ? "" : "s"}`;
    }

    if (setlistSongs.length === 0) {
      listEl.innerHTML = `<span class="setlist-empty-hint">Setlist is empty. Search a song and click "➕ Add to Set" to build your 4–5 song lineup!</span>`;
      return;
    }

    listEl.innerHTML = "";
    setlistSongs.forEach((song, idx) => {
      if (idx > 0) {
        const arrow = document.createElement("span");
        arrow.className = "setlist-transition-arrow";
        arrow.textContent = "➔";
        listEl.appendChild(arrow);
      }

      const pill = document.createElement("div");
      pill.className = `setlist-item-pill ${idx === activeSetlistIndex ? "active" : ""}`;
      pill.title = "Click to load and edit this song";
      pill.innerHTML = `
        <span class="setlist-item-num">#${idx + 1}</span>
        <span class="setlist-item-title">${song.title}</span>
        <span class="setlist-item-key">${song.key || "C"}</span>
        <span class="setlist-item-remove" title="Remove from set">✕</span>
      `;

      pill.addEventListener("click", (e) => {
        if (e.target.classList.contains("setlist-item-remove")) {
          e.stopPropagation();
          setlistSongs.splice(idx, 1);
          if (activeSetlistIndex >= setlistSongs.length) {
            activeSetlistIndex = setlistSongs.length - 1;
          }
          saveSetlistToStorage();
          return;
        }

        // Switch to this song in editor
        activeSetlistIndex = idx;
        renderSetlistTray();
        editor.value = song.chordpro_text;
        transposeSemitones = song.transpose_semitones || 0;
        lblTranspose.textContent = `${transposeSemitones > 0 ? "+" : ""}${transposeSemitones} st`;
        selCapo.value = String(song.capo || 0);
        chkCapoShapes.checked = song.use_capo_shapes !== false;
        refreshPreview();
      });

      listEl.appendChild(pill);
    });
  }

  async function exportFullSongbookPDF() {
    if (setlistSongs.length === 0) {
      alert("Please add at least 1 song to the Sunday Set before exporting!");
      return;
    }

    // Ensure currently edited song settings are refreshed in setlist if it's in the set
    if (activeSetlistIndex >= 0 && activeSetlistIndex < setlistSongs.length) {
      const curr = setlistSongs[activeSetlistIndex];
      curr.chordpro_text = editor.value;
      curr.transpose_semitones = transposeSemitones;
      curr.capo = parseInt(selCapo.value, 10) || 0;
      curr.key = boxOutputKey ? boxOutputKey.textContent.trim() : curr.key;
      saveSetlistToStorage();
    }

    const payload = {
      title: "Sunday Worship Setlist",
      songs: setlistSongs.map((s) => ({
        chordpro_text: s.chordpro_text,
        transpose_semitones: s.transpose_semitones,
        capo: s.capo,
        use_capo_shapes: s.use_capo_shapes,
        use_nashville: s.use_nashville,
        tone_style: s.tone_style || "marks",
        show_pinyin: s.show_pinyin !== false,
      })),
    };

    try {
      const res = await fetch("/api/export/songbook-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Failed to export setlist PDF");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "Sunday_Worship_Setlist.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Export error: " + e.message);
    }
  }

  document.getElementById("btn-add-to-setlist")?.addEventListener("click", addCurrentSongToSetlist);
  document.getElementById("btn-export-songbook")?.addEventListener("click", exportFullSongbookPDF);
  document.getElementById("btn-clear-setlist")?.addEventListener("click", () => {
    if (confirm("Clear all songs from the Sunday setlist?")) {
      setlistSongs = [];
      activeSetlistIndex = -1;
      saveSetlistToStorage();
    }
  });

  document.getElementById("btn-export-pdf")?.addEventListener("click", () =>
    downloadFile("/api/export/pdf", `${previewTitle.textContent || "score"}.pdf`)
  );
  document.getElementById("btn-export-xml")?.addEventListener("click", () =>
    downloadFile("/api/export/musicxml", `${previewTitle.textContent || "score"}.musicxml`)
  );
  document.getElementById("btn-export-cho")?.addEventListener("click", () =>
    downloadFile("/api/export/chordpro", `${previewTitle.textContent || "score"}.cho`)
  );

  // Initial load: start with Shout to the Lord and load existing setlist
  loadDemo("shout");
  loadSetlistFromStorage();
})();
