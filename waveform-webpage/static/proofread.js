import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'

document.addEventListener('DOMContentLoaded', function () {
    // --- 元素获取 ---
    const statusDisplay = document.getElementById('statusDisplay');
    const correctorUI = document.getElementById('corrector-ui');
    const videoElement = document.getElementById('videoPlayer');
    const currentSentenceTitle = document.getElementById('current-sentence-title');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const playSelectionBtn = document.getElementById('play-selection-btn');
    const confirmClipBtn = document.getElementById('confirm-clip-btn');
    const ignoreClipBtn = document.getElementById('ignore-clip-btn');
    const packageBtn = document.getElementById('package-btn');
    const packageContainer = document.getElementById('package-container');
    const sentenceSelectorCard = document.getElementById('sentence-selector-card');
    const sentenceList = document.getElementById('sentence-list');
    const manualUploadCard = document.getElementById('manual-upload-card');
    const loadManualBtn = document.getElementById('load-manual-files-btn');
    const manualVideoInput = document.getElementById('manualVideoFiles'); // 修改为多文件
    const manualJsonInput = document.getElementById('manualJsonFile');
    const uploadAnkiBtn = document.getElementById('upload-anki-btn'); // 新增

    // --- 全局状态变量 ---
    let wavesurfer = null;
    let regionsPlugin = null;
    let activeRegion = null;
    let analysisData = null; 
    let currentClipId = null;
    let isPlayingRegion = false;
    // --- 新增: 存储已确认的片段数据，用于 Anki 上传和打包 ---
    let confirmedClipsData = {};
    // --- 新增: 存储手动上传的视频文件引用 ---
    let manualVideoFilesMap = new Map();

    // --- 主初始化函数 ---
    async function init() {
        confirmClipBtn.addEventListener('click', handleConfirmClip);
        ignoreClipBtn.addEventListener('click', handleIgnoreClip);
        packageBtn.addEventListener('click', handlePackaging);
        uploadAnkiBtn.addEventListener('click', handleUploadToAnki); // 新增
        loadManualBtn.addEventListener('click', handleManualLoad);

        const params = new URLSearchParams(window.location.search);
        const jsonUrl = params.get('json_url');

        if (jsonUrl) {
            // 自动加载模式
            await loadDataAndSetupEditor(null, jsonUrl);
        } else {
            // 手动加载模式
            updateStatus('请手动上传文件。', 'info');
            manualUploadCard.classList.remove('hidden');
        }
    }

    // --- 数据加载与编辑器设置 (重构以支持多视频) ---
    async function loadDataAndSetupEditor(videoFilesMap, jsonUrlOrData) {
        try {
            updateStatus('正在加载分析结果...', 'loading');
            let jsonData;
            if (typeof jsonUrlOrData === 'string') {
                const response = await fetch(jsonUrlOrData);
                if (!response.ok) throw new Error(`无法加载JSON文件 (${response.status})`);
                jsonData = await response.json();
            } else {
                jsonData = jsonUrlOrData;
            }
            analysisData = jsonData;

            if (!analysisData.clips || analysisData.clips.length === 0) {
                throw new Error('分析结果中不包含任何有效的句子片段。');
            }
            
            // 如果是手动加载，使用传入的 videoFilesMap
            if (videoFilesMap) {
                manualVideoFilesMap = videoFilesMap;
            }

            analysisData.clips.forEach(clip => {
                clip.status = 'pending'; // pending, confirmed, ignored
                clip.proofed_start = clip.predicted_start;
                clip.proofed_end = clip.predicted_end;
            });
            
            // 初始化 WaveSurfer (但不加载媒体，等待用户选择)
            await setupWaveSurfer();
            
            populateSentenceSelector(analysisData.clips);
            sentenceSelectorCard.classList.remove('hidden');
            packageContainer.classList.remove('hidden');
            manualUploadCard.classList.add('hidden');
            updateStatus('初始化完成。请从下方列表选择一个句子开始校对。', 'success');

        } catch (error) {
            console.error('加载或设置时出错:', error);
            updateStatus(`加载数据失败: ${error.message}`, 'error');
            correctorUI.classList.add('hidden');
            sentenceSelectorCard.classList.add('hidden');
            packageContainer.classList.add('hidden');
            manualUploadCard.classList.remove('hidden');
        }
    }

    // --- 初始化 WaveSurfer (修改为不立即加载媒体) ---
    function setupWaveSurfer() {
        if (wavesurfer) wavesurfer.destroy();
        
        regionsPlugin = RegionsPlugin.create();
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgb(135, 168, 206)',
            progressColor: 'rgb(43, 83, 131)',
            media: videoElement,
            autoScroll: true,
            minPxPerSec: 100,
            plugins: [regionsPlugin, TimelinePlugin.create({ container: '#timeline' })],
        });

        // 事件监听保持不变
        wavesurfer.on('pause', () => {
            isPlayingRegion = false;
            playSelectionBtn.textContent = '▶️ 播放选区';
        });
        wavesurfer.on('play', () => {
            isPlayingRegion = true;
            playSelectionBtn.textContent = '⏸️ 暂停';
        });
        regionsPlugin.on('region-out', (region) => {
            if (isPlayingRegion) {
                wavesurfer.pause();
            }
        });
        playSelectionBtn.onclick = () => {
            if (!activeRegion) return;
            // 使用 activeRegion.play() 保证只播放选区
            if (wavesurfer.isPlaying()) {
                wavesurfer.pause();
            } else {
                activeRegion.play();
            }
        };
        return Promise.resolve();
    }
    
    // --- 填充句子选择列表 (无变动) ---
    function populateSentenceSelector(clips) {
        sentenceList.innerHTML = '';
        clips.forEach(clip => {
            const item = document.createElement('div');
            item.className = 'sentence-item';
            item.id = `item-${clip.id}`;
            item.innerHTML = `
                <span class="sentence-text">${clip.sentence}</span>
                <span class="sentence-status" id="status-${clip.id}"></span>
            `;
            item.addEventListener('click', () => {
                if (item.classList.contains('active')) return;
                document.querySelectorAll('.sentence-item.active').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                setupCorrectorForClip(clip);
            });
            sentenceList.appendChild(item);
            updateClipStatusIndicator(clip);
        });
    }

    // --- 为选定片段设置校对器 (核心重构) ---
    async function setupCorrectorForClip(clip) {
        if (wavesurfer.isPlaying()) wavesurfer.pause();
        
        currentClipId = clip.id;
        correctorUI.classList.remove('hidden');
        updateStatus(`正在加载视频: ${clip.original_video_filename}`, 'loading');
        currentSentenceTitle.textContent = `正在校对: "${clip.sentence}"`;
        
        // 检查是否需要切换视频源
        const currentSrc = videoElement.src.split('/').pop();
        const requiredSrcFilename = clip.video_url.split('/').pop();

        if (decodeURIComponent(currentSrc) !== decodeURIComponent(requiredSrcFilename)) {
            console.log(`需要切换视频源: 从 '${currentSrc}' 到 '${requiredSrcFilename}'`);
            let videoUrl;
            // 检查是自动加载模式还是手动加载模式
            if (manualVideoFilesMap.size > 0) {
                const file = manualVideoFilesMap.get(clip.original_video_filename);
                if (!file) {
                    updateStatus(`错误: 在手动上传的文件中找不到 ${clip.original_video_filename}`, 'error');
                    return;
                }
                videoUrl = URL.createObjectURL(file);
            } else {
                videoUrl = clip.video_url;
            }
            
            await wavesurfer.load(videoUrl);
            updateStatus(`视频 ${clip.original_video_filename} 加载完成`, 'success');
        }

        // 视频加载完成后再操作 Region
        regionsPlugin.clearRegions();
        activeRegion = regionsPlugin.addRegion({
            id: clip.id,
            start: clip.proofed_start,
            end: clip.proofed_end,
            color: 'rgba(246, 173, 85, 0.25)',
            drag: true,
            resize: true,
        });

        updateTimestampInputs();
        
        activeRegion.on('update-end', () => {
             const roundedStart = parseFloat(activeRegion.start.toFixed(3));
             const roundedEnd = parseFloat(activeRegion.end.toFixed(3));
             activeRegion.start = roundedStart;
             activeRegion.end = roundedEnd;
             clip.proofed_start = roundedStart;
             clip.proofed_end = roundedEnd;
             updateTimestampInputs();
        });
        
        const seekPosition = activeRegion.start / wavesurfer.getDuration();
        wavesurfer.seekTo(seekPosition);
    }
    
    // --- 核心操作逻辑 ---
    async function handleConfirmClip() {
        if (!currentClipId || !activeRegion) return;

        const clip = analysisData.clips.find(c => c.id === currentClipId);
        updateClipStatusIndicator(clip, 'loading');
        confirmClipBtn.disabled = true;

        const payload = {
            start_time: clip.proofed_start,
            end_time: clip.proofed_end,
            clip_source_path: clip.clip_source_path,
            original_video_filename: clip.original_video_filename,
            sentence_id: clip.id,
        };

        try {
            const clipResponse = await fetch('/clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!clipResponse.ok) {
                const err = await clipResponse.json();
                throw new Error(err.error || '后端裁剪失败');
            }
            const clipResult = await clipResponse.json();

            const audioResponse = await fetch(clipResult.download_url);
            if (!audioResponse.ok) throw new Error('下载裁剪后的音频失败');
            const audioBlob = await audioResponse.blob();

            // 将 Blob 转换为 Base64 字符串
            const audioBase64 = await blobToBase64(audioBlob);

            // 存储完整信息以备上传或打包
            confirmedClipsData[clip.id] = {
                note_id: clip.note_id,
                sentence: clip.sentence,
                audio_base64: audioBase64,
                audio_blob: audioBlob, // 保留 blob 用于打包
                original_video_filename: clip.original_video_filename,
                proofed_start: clip.proofed_start,
                proofed_end: clip.proofed_end,
            };
            
            clip.status = 'confirmed';
            updateClipStatusIndicator(clip);
            updateActionButtons();

        } catch (error) {
            console.error('确认裁剪时出错:', error);
            updateStatus(`错误: ${error.message}`, 'error');
            clip.status = 'pending';
            updateClipStatusIndicator(clip);
        } finally {
            confirmClipBtn.disabled = false;
        }
    }

    function handleIgnoreClip() {
        if (!currentClipId) return;
        const clip = analysisData.clips.find(c => c.id === currentClipId);
        clip.status = 'ignored';
        if (confirmedClipsData[clip.id]) {
            delete confirmedClipsData[clip.id];
        }
        updateClipStatusIndicator(clip);
        updateActionButtons();
    }
    
    // --- 新增: 上传到 Anki 的处理函数 ---
    async function handleUploadToAnki() {
        const clipsToUpload = Object.values(confirmedClipsData).map(data => ({
            note_id: data.note_id,
            sentence: data.sentence,
            audio_base64: data.audio_base64,
            original_video_filename: data.original_video_filename
        }));

        if (clipsToUpload.length === 0) {
            updateStatus('没有已确认的音频可供上传。', 'error');
            return;
        }
        
        // 检查是否有片段缺少 note_id
        const missingNoteIdCount = clipsToUpload.filter(c => !c.note_id).length;
        if (missingNoteIdCount > 0) {
            if (!confirm(`${missingNoteIdCount} 个已确认的片段缺少 Anki 笔记ID，它们将被跳过。要继续上传吗？`)) {
                return;
            }
        }

        updateStatus(`正在上传 ${clipsToUpload.length} 个音频到 Anki...`, 'loading');
        uploadAnkiBtn.disabled = true;
        packageBtn.disabled = true;

        try {
            const response = await fetch('/api/anki/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clips: clipsToUpload.filter(c => c.note_id) }) // 只发送有 note_id 的
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || '上传失败');
            }
            updateStatus(result.message || '成功上传至 Anki！', 'success');

        } catch (error) {
            console.error('上传到 Anki 时出错:', error);
            updateStatus(`上传 Anki 失败: ${error.message}`, 'error');
        } finally {
            updateActionButtons();
        }
    }

    async function handlePackaging() {
        const confirmedCount = Object.keys(confirmedClipsData).length;
        if (confirmedCount === 0) {
            updateStatus('没有已确认的音频可供打包。', 'error');
            return;
        }
        updateStatus('正在生成 ZIP 文件...', 'loading');
        packageBtn.disabled = true;
        uploadAnkiBtn.disabled = true;

        try {
            const zip = new JSZip();
            const mapping = [];
            
            for (const clipId in confirmedClipsData) {
                const clipData = confirmedClipsData[clipId];
                const baseName = clipData.original_video_filename.replace(/\.[^/.]+$/, "");
                const sanitize = (str) => str.replace(/[\\?/*:"<>|]/g, "").replace(/\s+/g, '_');
                const sentencePart = sanitize(clipData.sentence.substring(0, 20));
                const desiredFilename = `${baseName}_${sentencePart}.wav`;
                
                zip.file(desiredFilename, clipData.audio_blob);
                mapping.push({ filename: desiredFilename, sentence: clipData.sentence });
            }

            zip.file('mapping.json', JSON.stringify(mapping, null, 2));
            const zipBlob = await zip.generateAsync({ type: "blob", compression: "STORE" });
            const downloadLink = document.createElement('a');
            downloadLink.href = URL.createObjectURL(zipBlob);
            downloadLink.download = `audio_package.zip`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            updateStatus(`成功生成包含 ${confirmedCount} 个音频的压缩包。`, 'success');
        } catch (error) {
            console.error('打包时出错:', error);
            updateStatus(`打包失败: ${error.message}`, 'error');
        } finally {
            updateActionButtons();
        }
    }

    function handleManualLoad() {
        const videoFiles = manualVideoInput.files;
        const jsonFile = manualJsonInput.files[0];
        if (videoFiles.length === 0 || !jsonFile) {
            updateStatus('请同时选择视频文件和JSON文件。', 'error');
            return;
        }
        // 创建一个文件名到 File 对象的映射
        const videoMap = new Map();
        for (const file of videoFiles) {
            videoMap.set(file.name, file);
        }

        const jsonReader = new FileReader();
        jsonReader.onload = (event) => {
            try {
                const jsonData = JSON.parse(event.target.result);
                loadDataAndSetupEditor(videoMap, jsonData);
            } catch (e) {
                updateStatus(`手动加载JSON失败: ${e.message}`, 'error');
            }
        };
        jsonReader.readAsText(jsonFile);
    }

    // --- 辅助函数 ---
    function updateClipStatusIndicator(clip, tempStatus = null) {
        const indicator = document.getElementById(`status-${clip.id}`);
        if (!indicator) return;
        
        const status = tempStatus || clip.status;

        switch (status) {
            case 'loading':
                indicator.textContent = '...';
                indicator.className = 'sentence-status loading';
                break;
            case 'confirmed':
                indicator.textContent = '✔ 已确认';
                indicator.className = 'sentence-status confirmed';
                break;
            case 'ignored':
                indicator.textContent = '❌ 已忽略';
                indicator.className = 'sentence-status ignored';
                break;
            default: // pending
                indicator.textContent = '';
                indicator.className = 'sentence-status';
        }
    }

    function updateTimestampInputs() {
        if (activeRegion) {
            startTimeInput.value = activeRegion.start.toFixed(3);
            endTimeInput.value = activeRegion.end.toFixed(3);
        }
    }

    function updateActionButtons() {
        const count = Object.keys(confirmedClipsData).length;
        packageBtn.textContent = `📦 生成并下载 ZIP (${count} 个文件)`;
        uploadAnkiBtn.textContent = `🚀 上传至 Anki (${count} 个文件)`;
        packageBtn.disabled = count === 0;
        uploadAnkiBtn.disabled = count === 0;
    }

    function updateStatus(message, type = 'info') {
        statusDisplay.textContent = message;
        statusDisplay.className = 'status-info';
        if (type === 'success') statusDisplay.style.backgroundColor = '#c6f6d5';
        else if (type === 'error') statusDisplay.style.backgroundColor = '#fed7d7';
        else if (type === 'loading') statusDisplay.style.backgroundColor = '#bee3f8';
        else statusDisplay.style.backgroundColor = '#f7fafc';
    }
    
    function blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                // The result includes the data URL prefix, so we need to remove it.
                // e.g., "data:audio/wav;base64,UklGRiYAA..." -> "UklGRiYAA..."
                const base64String = reader.result.split(',')[1];
                resolve(base64String);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    // --- 功能：允许鼠标滚轮在波形图上水平滚动 ---
    // 也就是，不需要 Shift + 滚轮，即可自由滚动
    const waveForm = document.getElementById('waveform');

    // 为该元素添加 'wheel' 事件监听器
    waveForm.addEventListener('wheel', (event) => {
        try {
            // 找到 Shadow DOM 的宿主 (host)
            // 根据 WaveSurfer 的结构，它是 #waveform 里的第一个 <div>
            const shadowHost = waveForm.querySelector('div');

            // 访问 Shadow DOM
            if (shadowHost && shadowHost.shadowRoot) {
                const shadowRoot = shadowHost.shadowRoot;

                // 在 Shadow DOM 内部查找真正的滚动元素
                // 它的 class 是 "scroll"
                const scrollableElement = shadowRoot.querySelector('.scroll');

                if (scrollableElement) {
                    // 检查内容是否真的需要滚动
                    if (scrollableElement.scrollWidth > scrollableElement.clientWidth) {
                        // 阻止页面默认的垂直滚动行为
                        event.preventDefault();

                        // 将滚轮的垂直偏移量应用到目标的水平滚动上
                        scrollableElement.scrollLeft += event.deltaY;
                    }
                }
            }
        } catch (error) {
            // 如果发生任何错误，在控制台打印出来，方便调试
            console.error('Error during horizontal scroll:', error);
        }
    }, { passive: false }); // 使用 { passive: false } 来确保 preventDefault() 生效

    init();
});
