// ─────────────────────────────────────────────────────────────────────────────
// NHS 3-Model Ensemble Clinical Coding UI
// Full entity extraction: Diagnoses (3-model), Symptoms, Procedures, Medications, Vitals
// ─────────────────────────────────────────────────────────────────────────────

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const excelFileSelect    = document.getElementById('excel-file-select');
const columnSelect       = document.getElementById('column-select');
const loadFileBtn        = document.getElementById('load-file-btn');
const recordsList        = document.getElementById('records-list');
const reviewedCountEl    = document.getElementById('reviewed-count');

const recordIndexBadge   = document.getElementById('record-index-badge');
const recordProgressText = document.getElementById('record-progress-text');
const prevRecordBtn      = document.getElementById('prev-record-btn');
const nextRecordBtn      = document.getElementById('next-record-btn');

const clinicalLetterText = document.getElementById('clinical-letter-text');
const highlightToggle    = document.getElementById('highlight-toggle');
const runExtractionBtn   = document.getElementById('run-extraction-btn');
const autoApproveBtn     = document.getElementById('auto-approve-btn');

const extractionEmptyState = document.getElementById('extraction-empty-state');
const entitiesWorkspace  = document.getElementById('entities-workspace');
const entitiesList       = document.getElementById('entities-list');
const categoryTabs       = document.getElementById('category-tabs');

const codingFooter       = document.getElementById('coding-footer');
const classifySnomedBtn  = document.getElementById('classify-snomed-btn');
const saveRowBtn         = document.getElementById('save-row-btn');
const exportBtn          = document.getElementById('export-btn');

const snomedModal        = document.getElementById('snomed-modal');
const snomedSearchInput  = document.getElementById('snomed-search-input');
const snomedSearchSubmit = document.getElementById('snomed-search-submit');
const snomedModalCategory = document.getElementById('snomed-modal-category');
const snomedResultsContainer = document.getElementById('snomed-results-container');

const toast = document.getElementById('toast');
const toastTitle = document.getElementById('toast-title');
const toastDesc = document.getElementById('toast-desc');
const toastIcon = document.getElementById('toast-icon');

// ── State ────────────────────────────────────────────────────────────────────
let loadedRecords = [];
let currentRecordIndex = 0;
let currentRecord = null;
let activeEntityDecisions = [];   // { id, text, category, confidence, validation_status, status, snomed, decision }
let activeCategoryFilter = 'All';
let highlightEnabled = true;
let snomedTargetEntityId = null;

// ── Status options per category ───────────────────────────────────────────────
const STATUS_OPTIONS = {
    Diagnosis:  ['Current', 'Historical', 'Negated', 'Resolved', 'Suspected'],
    Symptom:    ['Current', 'Historical', 'Negated', 'Resolved'],
    Procedure:  ['Performed', 'Planned', 'Recommended', 'Monitoring'],
    Medication: ['Current', 'Started', 'Stopped', 'Changed', 'Recommended'],
    Vital:      ['N/A']
};

function getStatusOptions(category, selected) {
    const options = STATUS_OPTIONS[category] || ['Current'];
    return options.map(o =>
        `<option value="${o}"${o === selected ? ' selected' : ''}>${o}</option>`
    ).join('');
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    fetchAvailableFiles();
    attachEventListeners();
});

function attachEventListeners() {
    excelFileSelect.addEventListener('change', onFileSelected);
    loadFileBtn.addEventListener('click', onLoadRecords);
    prevRecordBtn.addEventListener('click', () => navigateRecord(-1));
    nextRecordBtn.addEventListener('click', () => navigateRecord(1));
    runExtractionBtn.addEventListener('click', runExtraction);
    autoApproveBtn.addEventListener('click', autoApproveHighConfidence);
    classifySnomedBtn.addEventListener('click', runClassifySnomed);
    saveRowBtn.addEventListener('click', saveAndNext);
    exportBtn.addEventListener('click', exportExcel);
    highlightToggle.addEventListener('click', toggleHighlight);

    // Category tabs
    categoryTabs.querySelectorAll('.cat-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            activeCategoryFilter = tab.dataset.cat;
            categoryTabs.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderEntityCards();
        });
    });

    // SNOMED modal
    snomedSearchSubmit.addEventListener('click', searchSnomedTerm);
    snomedSearchInput.addEventListener('keydown', e => { if (e.key === 'Enter') searchSnomedTerm(); });
    snomedModal.addEventListener('click', e => { if (e.target === snomedModal) closeSnomedModal(); });
}

// ── File Loading ──────────────────────────────────────────────────────────────
async function fetchAvailableFiles() {
    try {
        const res = await fetch('/api/files');
        const files = await res.json();
        excelFileSelect.innerHTML = '<option value="">-- Choose Excel File --</option>';
        files.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.path;
            opt.textContent = f.name;
            excelFileSelect.appendChild(opt);
        });
    } catch(e) {
        console.error('Failed to fetch files', e);
    }
}

async function onFileSelected() {
    const filePath = excelFileSelect.value;
    if (!filePath) {
        columnSelect.disabled = true;
        loadFileBtn.disabled = true;
        return;
    }
    try {
        const res = await fetch(`/api/columns?file_path=${encodeURIComponent(filePath)}`);
        const cols = await res.json();
        columnSelect.innerHTML = '';
        cols.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c;
            opt.textContent = c;
            columnSelect.appendChild(opt);
        });
        columnSelect.disabled = false;
        loadFileBtn.disabled = false;
    } catch(e) {
        showToast('Error', 'Could not read file columns.', 'danger');
    }
}

async function onLoadRecords() {
    const filePath = excelFileSelect.value;
    const column = columnSelect.value;
    if (!filePath) return;

    loadFileBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    loadFileBtn.disabled = true;
    try {
        const res = await fetch('/api/load-records', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ file_path: filePath, column })
        });
        const data = await res.json();
        if (data.success) {
            loadedRecords = data.records;
            currentRecordIndex = data.current_index || 0;
            renderRecordsSidebar();
            exportBtn.disabled = false;
            loadRecord(currentRecordIndex);
            showToast('Loaded', `${data.total_records} records loaded.`, 'success');
        }
    } catch(e) {
        showToast('Error', 'Failed to load records.', 'danger');
    } finally {
        loadFileBtn.innerHTML = '<i class="fa-solid fa-sync"></i> Load Records';
        loadFileBtn.disabled = false;
    }
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function renderRecordsSidebar() {
    const reviewed = loadedRecords.filter(r => r.reviewed).length;
    reviewedCountEl.textContent = `Reviewed: ${reviewed}/${loadedRecords.length}`;
    recordsList.innerHTML = '';
    loadedRecords.forEach((r, i) => {
        const div = document.createElement('div');
        div.className = `record-list-item${i === currentRecordIndex ? ' active' : ''}`;
        div.innerHTML = `
            <span class="record-list-num">#${i + 1}</span>
            <span class="record-list-status">${r.reviewed ? '<i class="fa-solid fa-check-circle text-success"></i>' : (r.has_extractions ? '<i class="fa-solid fa-circle-dot text-warning"></i>' : '<i class="fa-regular fa-circle text-muted"></i>')}</span>
        `;
        div.addEventListener('click', () => loadRecord(i));
        recordsList.appendChild(div);
    });
}

// ── Record Loading ────────────────────────────────────────────────────────────
async function loadRecord(index) {
    currentRecordIndex = index;
    const res = await fetch(`/api/record/${index}`);
    currentRecord = await res.json();

    recordIndexBadge.textContent = `Record #${index + 1}`;
    recordProgressText.textContent = `${index + 1} of ${loadedRecords.length}`;
    prevRecordBtn.disabled = index === 0;
    nextRecordBtn.disabled = index >= loadedRecords.length - 1;

    renderLetterText(currentRecord.text, []);
    highlightEnabled = true;
    highlightToggle.classList.add('active');

    // Reset entity workspace
    extractionEmptyState.classList.remove('hide');
    entitiesWorkspace.classList.add('hide');
    codingFooter.classList.add('hide');
    runExtractionBtn.disabled = false;
    autoApproveBtn.disabled = true;
    activeEntityDecisions = [];

    if (currentRecord.extracted_entities && currentRecord.extracted_entities.length > 0) {
        activeEntityDecisions = currentRecord.extracted_entities;
        showEntitiesWorkspace();
    }

    renderRecordsSidebar();
}

function renderLetterText(text, highlights) {
    if (!text) {
        clinicalLetterText.innerHTML = '<em style="color:var(--text-muted)">No letter content available.</em>';
        return;
    }
    if (!highlightEnabled || !highlights.length) {
        clinicalLetterText.textContent = text;
        return;
    }
    let html = escapeHtml(text);
    highlights.forEach(h => {
        if (h && h.length > 2) {
            const escaped = escapeHtml(h).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${escaped})`, 'gi');
            html = html.replace(regex, '<mark class="highlight">$1</mark>');
        }
    });
    clinicalLetterText.innerHTML = html;
}

function toggleHighlight() {
    highlightEnabled = !highlightEnabled;
    highlightToggle.classList.toggle('active', highlightEnabled);
    refreshHighlights();
}

function refreshHighlights() {
    const terms = activeEntityDecisions
        .filter(d => d.decision === 'Yes')
        .map(d => d.text);
    renderLetterText(currentRecord?.text || '', terms);
}

// ── Extraction ────────────────────────────────────────────────────────────────
async function runExtraction() {
    runExtractionBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running 3-Model Extraction...';
    runExtractionBtn.disabled = true;
    try {
        const res = await fetch(`/api/extract/${currentRecordIndex}`, { method: 'POST' });
        const data = await res.json();
        if (data.entities !== undefined) {
            activeEntityDecisions = data.entities;
            // Update sidebar
            if (loadedRecords[currentRecordIndex]) {
                loadedRecords[currentRecordIndex].has_extractions = true;
            }
            showEntitiesWorkspace();
            showToast('Extraction Complete', `${data.entities.length} entities extracted.`, 'success');
        }
    } catch(e) {
        showToast('Error', 'Entity extraction failed.', 'danger');
    } finally {
        runExtractionBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Entity Extraction';
        runExtractionBtn.disabled = false;
    }
}

function showEntitiesWorkspace() {
    extractionEmptyState.classList.add('hide');
    entitiesWorkspace.classList.remove('hide');
    codingFooter.classList.remove('hide');
    autoApproveBtn.disabled = false;
    activeCategoryFilter = 'All';
    categoryTabs.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
    categoryTabs.querySelector('[data-cat="All"]').classList.add('active');
    renderEntityCards();
    refreshHighlights();
}

// ── Entity Cards ──────────────────────────────────────────────────────────────
function renderEntityCards() {
    entitiesList.innerHTML = '';
    const filtered = activeCategoryFilter === 'All'
        ? [...activeEntityDecisions]
        : activeEntityDecisions.filter(e => e.category === activeCategoryFilter);

    // Sort entities by confidence descending (items with more confidence come first)
    filtered.sort((a, b) => (b.confidence || 0) - (a.confidence || 0) || (a.start || 0) - (b.start || 0));

    if (!filtered.length) {
        entitiesList.innerHTML = `<div class="list-empty-msg" style="text-align:center;padding:30px 0;color:var(--text-muted);">No ${activeCategoryFilter !== 'All' ? activeCategoryFilter : ''} entities found.</div>`;
        return;
    }

    filtered.forEach(ent => {
        const card = buildEntityCard(ent);
        entitiesList.appendChild(card);
    });
}

function buildEntityCard(ent) {
    const confCls = ent.confidence >= 0.80 ? 'confidence-high' : (ent.confidence >= 0.60 ? 'confidence-medium' : 'confidence-low');
    const confIcon = ent.confidence >= 0.80 ? 'fa-check-double' : (ent.confidence >= 0.60 ? 'fa-check' : 'fa-question');
    const pct = Math.round(ent.confidence * 100);

    const needsSnomed = ['Diagnosis', 'Symptom', 'Procedure'].includes(ent.category);
    const snomedSection = needsSnomed ? `
        <div class="snomed-link-row">
            <label><i class="fa-solid fa-code-branch"></i> SNOMED CT Code</label>
            <div class="snomed-field-wrapper">
                <input type="text" id="snomed-input-${ent.id}" class="form-control"
                    placeholder="Click search to look up SNOMED..."
                    value="${escapeHtml(ent.snomed || '')}"
                    onchange="updateEntityField('${ent.id}', 'snomed', this.value)">
                <button class="btn btn-xs btn-outline" onclick="openSnomedSearch('${ent.id}', '${ent.category}')">
                    <i class="fa-solid fa-magnifying-glass"></i>
                </button>
            </div>
        </div>` : '';

    const div = document.createElement('div');
    div.className = 'entity-card';
    div.dataset.id = ent.id;
    div.dataset.cat = ent.category;
    div.innerHTML = `
        <div class="entity-card-header">
            <div class="entity-title-row">
                <input type="text" class="entity-name-input" value="${escapeHtml(ent.text)}"
                    onchange="updateEntityField('${ent.id}', 'text', this.value)">
                <div class="entity-meta-row">
                    <span class="cat-badge cat-badge-${ent.category}">${ent.category}</span>
                    <span class="confidence-gauge ${confCls}">
                        <i class="fa-solid ${confIcon}"></i> ${pct}%
                    </span>
                    <span class="status-pill" style="font-size:10px;background:rgba(15,23,42,0.05);padding:2px 6px;border-radius:3px;">${escapeHtml(ent.validation_status || '')}</span>
                </div>
            </div>
        </div>

        <div class="decision-bar">
            <label>Coding Decision</label>
            <div class="btn-group-decision">
                <button class="btn-decision ${ent.decision === 'Yes' ? 'active-yes' : ''}"
                    onclick="setDecision('${ent.id}', 'Yes')">
                    <i class="fa-solid fa-check"></i> Include
                </button>
                <button class="btn-decision ${ent.decision === 'Do Not Need' ? 'active-skip' : ''}"
                    onclick="setDecision('${ent.id}', 'Do Not Need')">
                    <i class="fa-solid fa-minus"></i> Skip
                </button>
                <button class="btn-decision ${ent.decision === 'No' ? 'active-no' : ''}"
                    onclick="setDecision('${ent.id}', 'No')">
                    <i class="fa-solid fa-xmark"></i> Reject
                </button>
            </div>
        </div>

        <div class="entity-details-expansion${ent.decision === 'No' ? ' hide' : ''}">
            <div class="entity-detail-item">
                <label>Clinical Status</label>
                <select id="status-sel-${ent.id}" onchange="updateEntityField('${ent.id}', 'status', this.value)">
                    ${getStatusOptions(ent.category, ent.status)}
                </select>
            </div>
            <div class="entity-detail-item">
                <label>Category</label>
                <select id="cat-sel-${ent.id}" onchange="updateEntityCategory('${ent.id}', this.value)">
                    ${['Diagnosis','Symptom','Procedure','Medication','Vital'].map(c =>
                        `<option value="${c}"${c === ent.category ? ' selected':''}>${c}</option>`
                    ).join('')}
                </select>
            </div>
            ${snomedSection}
        </div>
    `;
    return div;
}

// ── Entity State Updates ──────────────────────────────────────────────────────
function setDecision(entityId, decision) {
    const ent = activeEntityDecisions.find(e => e.id === entityId);
    if (!ent) return;
    ent.decision = decision;

    const card = document.querySelector(`.entity-card[data-id="${entityId}"]`);
    if (!card) return;

    // Update button active states
    card.querySelectorAll('.btn-decision').forEach(btn => {
        btn.classList.remove('active-yes', 'active-no', 'active-skip');
    });
    const [yesBtn, skipBtn, noBtn] = card.querySelectorAll('.btn-decision');
    if (decision === 'Yes') yesBtn.classList.add('active-yes');
    else if (decision === 'Do Not Need') skipBtn.classList.add('active-skip');
    else if (decision === 'No') noBtn.classList.add('active-no');

    // Show/hide details expansion
    const expansion = card.querySelector('.entity-details-expansion');
    if (expansion) expansion.classList.toggle('hide', decision === 'No');

    refreshHighlights();
}

function updateEntityField(entityId, field, value) {
    const ent = activeEntityDecisions.find(e => e.id === entityId);
    if (ent) ent[field] = value;
}

function updateEntityCategory(entityId, newCategory) {
    const ent = activeEntityDecisions.find(e => e.id === entityId);
    if (!ent) return;
    ent.category = newCategory;
    ent.status = { Diagnosis:'Current', Symptom:'Current', Procedure:'Performed', Medication:'Current', Vital:'N/A' }[newCategory] || 'Current';

    // Re-render just that card
    const card = document.querySelector(`.entity-card[data-id="${entityId}"]`);
    if (card) {
        const newCard = buildEntityCard(ent);
        card.parentNode.replaceChild(newCard, card);
    }
}

// ── Auto Approve ──────────────────────────────────────────────────────────────
function autoApproveHighConfidence() {
    activeEntityDecisions.forEach(ent => {
        const target = ent.confidence >= 0.80 ? 'Yes' : 'Do Not Need';
        setDecision(ent.id, target);
    });
    showToast('Auto-coded', 'High confidence entities included, others skipped.', 'success');
}

// ── Classify & SNOMED ─────────────────────────────────────────────────────────
async function runClassifySnomed() {
    classifySnomedBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving & Mapping...';
    classifySnomedBtn.disabled = true;
    try {
        const payload = {
            row_index: currentRecordIndex,
            clinical_text: currentRecord.text,
            decisions: activeEntityDecisions
        };
        const res = await fetch('/api/process-row', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            // Update SNOMED fields in the UI from backend response
            data.processed_entities.forEach(pe => {
                if (pe.decision !== 'Yes') return;
                const ent = activeEntityDecisions.find(e => e.id === pe.id);
                if (ent) { ent.snomed = pe.snomed; ent.status = pe.status; }
                const snomedInput = document.getElementById(`snomed-input-${pe.id}`);
                if (snomedInput) snomedInput.value = pe.snomed || '';
            });
            if (loadedRecords[currentRecordIndex]) {
                loadedRecords[currentRecordIndex].reviewed = true;
            }
            renderRecordsSidebar();
            showToast('Mapped', 'SNOMED codes fetched and saved.', 'success');
        }
    } catch(e) {
        showToast('Error', 'Classification/SNOMED mapping failed.', 'danger');
    } finally {
        classifySnomedBtn.innerHTML = '<i class="fa-solid fa-gears"></i> Classify & Map SNOMED';
        classifySnomedBtn.disabled = false;
    }
}

// ── Save & Next ───────────────────────────────────────────────────────────────
async function saveAndNext() {
    saveRowBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';
    saveRowBtn.disabled = true;
    try {
        const payload = {
            row_index: currentRecordIndex,
            clinical_text: currentRecord.text,
            decisions: activeEntityDecisions
        };
        const res = await fetch('/api/process-row', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            if (loadedRecords[currentRecordIndex]) loadedRecords[currentRecordIndex].reviewed = true;
            renderRecordsSidebar();
            showToast('Saved', `Record #${currentRecordIndex + 1} saved.`, 'success');
            if (currentRecordIndex < loadedRecords.length - 1) {
                setTimeout(() => loadRecord(currentRecordIndex + 1), 400);
            }
        }
    } catch(e) {
        showToast('Error', 'Failed to save.', 'danger');
    } finally {
        saveRowBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save & Next';
        saveRowBtn.disabled = false;
    }
}

function navigateRecord(offset) {
    const next = currentRecordIndex + offset;
    if (next >= 0 && next < loadedRecords.length) loadRecord(next);
}

// ── Export ────────────────────────────────────────────────────────────────────
async function exportExcel() {
    exportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Exporting...';
    exportBtn.disabled = true;
    try {
        const res = await fetch('/api/export', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            const link = document.createElement('a');
            link.href = data.download_url;
            link.setAttribute('download', data.filename);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast('Exported', 'Spreadsheet downloaded successfully.', 'success');
        }
    } catch(e) {
        showToast('Error', 'Export failed.', 'danger');
    } finally {
        exportBtn.innerHTML = '<i class="fa-solid fa-file-export"></i> Export Final Spreadsheet';
        exportBtn.disabled = false;
    }
}

// ── SNOMED Modal ──────────────────────────────────────────────────────────────
window.openSnomedSearch = function(entityId, category) {
    snomedTargetEntityId = entityId;
    snomedModalCategory.textContent = category;
    snomedSearchInput.value = '';
    snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg">Type a search term and click search.</div>';
    snomedModal.classList.remove('hide');
    snomedSearchInput.focus();
};

window.closeSnomedModal = function() {
    snomedModal.classList.add('hide');
    snomedTargetEntityId = null;
};

async function searchSnomedTerm() {
    const term = snomedSearchInput.value.trim();
    if (term.length < 2) {
        snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg" style="color:var(--danger)">Enter at least 2 characters.</div>';
        return;
    }
    snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg"><i class="fa-solid fa-spinner fa-spin"></i> Querying FHIR servers...</div>';
    try {
        const category = snomedModalCategory.textContent;
        const res = await fetch(`/api/snomed-search?term=${encodeURIComponent(term)}&category=${encodeURIComponent(category)}`);
        const data = await res.json();
        if (data.success && data.results && data.results.length > 0) {
            snomedResultsContainer.innerHTML = '';
            data.results.forEach(item => {
                const div = document.createElement('div');
                div.className = 'snomed-result-item';
                div.innerHTML = `
                    <div class="snomed-result-info">
                        <span class="snomed-result-display">${escapeHtml(item.snomed_display)}</span>
                        <span class="snomed-result-code">Code: ${item.snomed_code}</span>
                    </div>
                    <button class="btn btn-xs btn-primary">Select</button>
                `;
                div.addEventListener('click', () => selectSnomedCode(item.snomed_code, item.snomed_display));
                snomedResultsContainer.appendChild(div);
            });
        } else {
            snomedResultsContainer.innerHTML = `<div class="snomed-empty-msg">No concepts found for "${escapeHtml(term)}".</div>`;
        }
    } catch(e) {
        snomedResultsContainer.innerHTML = '<div class="snomed-empty-msg" style="color:var(--danger)">Error querying FHIR server.</div>';
    }
}

function selectSnomedCode(code, display) {
    if (snomedTargetEntityId) {
        const ent = activeEntityDecisions.find(e => e.id === snomedTargetEntityId);
        const formatted = `${code} | ${display}`;
        if (ent) ent.snomed = formatted;
        const inp = document.getElementById(`snomed-input-${snomedTargetEntityId}`);
        if (inp) inp.value = formatted;
    }
    closeSnomedModal();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(title, desc, type = 'success') {
    toastTitle.textContent = title;
    toastDesc.textContent = desc;
    if (type === 'danger') {
        toastIcon.className = 'fa-solid fa-circle-exclamation';
        toastIcon.style.color = 'var(--danger)';
        toast.style.borderColor = 'rgba(239,68,68,0.4)';
    } else {
        toastIcon.className = 'fa-solid fa-circle-check';
        toastIcon.style.color = 'var(--success)';
        toast.style.borderColor = 'rgba(16,185,129,0.4)';
    }
    toast.classList.remove('hide');
    setTimeout(() => toast.classList.add('hide'), 4000);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
