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
    const manualVideoInput = document.getElementById('manualVideoFile');
    const manualJsonInput = document.getElementById('manualJsonFile');

    // --- 全局状态变量 ---
    let wavesurfer = null;
    let regionsPlugin = null;
    let activeRegion = null;
    let analysisData = null; // 存储从JSON加载的原始数据
    let currentClipId = null;
    let isAudioUnlocked = false;
    
    // 【功能恢复】严格按照原始文件，恢复此状态变量
    let isPlayingRegion = false;
    
    // confirmedClipsData: 暂存所有已确认的音频片段，用于最终打包
    let confirmedClipsData = {};

    // --- 主初始化函数 ---
    async function init() {
        // 绑定事件
        confirmClipBtn.addEventListener('click', handleConfirmClip);
        ignoreClipBtn.addEventListener('click', handleIgnoreClip);
        packageBtn.addEventListener('click', handlePackaging);
        loadManualBtn.addEventListener('click', handleManualLoad);

        // 尝试从URL参数自动加载
        const params = new URLSearchParams(window.location.search);
        const videoUrl = params.get('video_url');
        const jsonUrl = params.get('json_url');
        if (videoUrl && jsonUrl) {
            await loadDataAndSetupEditor(videoUrl, jsonUrl);
        } else {
            updateStatus('错误：缺少文件链接。请手动上传文件。', 'error');
            manualUploadCard.classList.remove('hidden');
        }
    }

    // --- 数据加载与编辑器设置 ---
    async function loadDataAndSetupEditor(videoUrl, jsonUrl) {
        try {
            updateStatus('正在加载分析结果...', 'loading');
            let jsonData;
            if (typeof jsonUrl === 'string') { // URL
                const response = await fetch(jsonUrl);
                if (!response.ok) throw new Error(`无法加载JSON文件 (${response.status})`);
                jsonData = await response.json();
            } else { // File Object
                jsonData = jsonUrl;
            }
            analysisData = jsonData;

            if (!analysisData.clips || analysisData.clips.length === 0) {
                throw new Error('分析结果中不包含任何有效的句子片段。');
            }

            // 为每个 clip 添加校对状态和暂存的校对后时间
            analysisData.clips.forEach(clip => {
                clip.status = 'pending'; // 'pending', 'confirmed', 'ignored'
                clip.proofed_start = clip.predicted_start;
                clip.proofed_end = clip.predicted_end;
            });

            updateStatus('数据加载成功，正在初始化播放器...', 'loading');
            await setupWaveSurfer(videoUrl);
            
            populateSentenceSelector(analysisData.clips);
            sentenceSelectorCard.classList.remove('hidden');
            packageContainer.classList.remove('hidden');
            manualUploadCard.classList.add('hidden');
            updateStatus('初始化完成。请从下方列表选择一个句子开始校对。', 'success');
        } catch (error) {
            console.error('加载或设置时出错:', error);
            updateStatus(`加载数据失败: ${error.message}`, 'error');
            manualUploadCard.classList.remove('hidden');
        }
    }

    // --- 初始化 WaveSurfer ---
    function setupWaveSurfer(mediaUrl) {
        return new Promise((resolve, reject) => {
            if (wavesurfer) wavesurfer.destroy();
            videoElement.src = mediaUrl;
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

            // 【功能恢复】严格按照原始文件，恢复播放相关的事件监听
            // 监听播放器的 'pause' 事件来更新状态和按钮文本
            wavesurfer.on('pause', () => {
                isPlayingRegion = false;
                playSelectionBtn.textContent = '▶️ 播放选区';
            });
            
            // 监听播放器的 'play' 事件来更新状态和按钮文本
            wavesurfer.on('play', () => {
                isPlayingRegion = true;
                playSelectionBtn.textContent = '⏸️ 暂停';
            });

            // 【功能恢复】严格按照原始文件，恢复播放超出选区时自动暂停的核心功能
            regionsPlugin.on('region-out', () => {
                if (isPlayingRegion) {
                    wavesurfer.pause();
                }
            });

            // 【功能恢复】严格按照原始文件，恢复播放按钮的点击逻辑
            playSelectionBtn.onclick = () => {
                if (!activeRegion) return;
                if (wavesurfer.isPlaying()) {
                    wavesurfer.pause();
                } else {
                    activeRegion.play();
                }
            };

            wavesurfer.on('ready', resolve);
            wavesurfer.on('error', reject);
        });
    }

    // --- 填充句子选择列表 ---
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
            updateClipStatusIndicator(clip); // 初始化状态显示
        });
    }

    // --- 为选定片段设置校对器 ---
    function setupCorrectorForClip(clip) {
        if (wavesurfer.isPlaying()) wavesurfer.pause();

        currentClipId = clip.id;
        correctorUI.classList.remove('hidden');
        currentSentenceTitle.textContent = `正在校对: "${clip.sentence}"`;
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
        
        // 监听选区更新，实时暂存时间
        activeRegion.on('update-end', () => {
             clip.proofed_start = activeRegion.start;
             clip.proofed_end = activeRegion.end;
             updateTimestampInputs();
        });
        
        const seekPosition = activeRegion.start / wavesurfer.getDuration();
        wavesurfer.seekTo(seekPosition);

        // 解锁音频播放
        if (!isAudioUnlocked) {
            wavesurfer.play().then(() => {
                wavesurfer.pause();
                wavesurfer.seekTo(seekPosition);
            }).catch(console.error);
            isAudioUnlocked = true;
        }
    }

    // --- 核心操作逻辑 ---
    async function handleConfirmClip() {
        if (!currentClipId || !activeRegion) return;

        const clip = analysisData.clips.find(c => c.id === currentClipId);
        const statusIndicator = document.getElementById(`status-${clip.id}`);
        if (statusIndicator) statusIndicator.textContent = '裁剪中...';
        confirmClipBtn.disabled = true;

        const payload = {
            start_time: clip.proofed_start,
            end_time: clip.proofed_end,
            clip_source_path: analysisData.clip_source_path,
            original_video_filename: analysisData.original_video_filename,
            sentence_id: currentClipId,
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

            confirmedClipsData[clip.id] = {
                sentence: clip.sentence,
                audioBlob: audioBlob,
            };
            clip.status = 'confirmed';
            updateClipStatusIndicator(clip);
            updatePackageButton();

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
        updatePackageButton();
    }
    
    async function handlePackaging() {
        if (Object.keys(confirmedClipsData).length === 0) {
            updateStatus('没有已确认的音频可供打包。', 'error');
            return;
        }

        updateStatus('正在生成 ZIP 文件，请稍候...', 'loading');
        packageBtn.disabled = true;

        try {
            const zip = new JSZip();
            const mapping = [];
            const videoBaseName = analysisData.original_video_filename.replace(/\.[^/.]+$/, "");

            // Helper to sanitize filenames
            const sanitize = (str) => str.replace(/[\\?/*:"<>|]/g, "").replace(/\s+/g, '_');

            for (const clipId in confirmedClipsData) {
                const clipData = confirmedClipsData[clipId];
                // Find the original clip object to get the proofed times
                const originalClip = analysisData.clips.find(c => c.id === clipId);
                if (!originalClip) continue;

                const sentencePart = sanitize(clipData.sentence.substring(0, 20)); // Sanitize and limit length
                const timeRangePart = `${originalClip.proofed_start.toFixed(2)}s-${originalClip.proofed_end.toFixed(2)}s`;
                
                const desiredFilename = `${videoBaseName}_${sentencePart}_${timeRangePart}.wav`;

                zip.file(desiredFilename, clipData.audioBlob);
                mapping.push({
                    filename: desiredFilename,
                    sentence: clipData.sentence
                });
            }

            zip.file('mapping.json', JSON.stringify(mapping, null, 2));

            const zipBlob = await zip.generateAsync({
                type: "blob",
                compression: "STORE"
            });

            const downloadLink = document.createElement('a');
            downloadLink.href = URL.createObjectURL(zipBlob);
            downloadLink.download = `${videoBaseName}_audio_package.zip`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            updateStatus(`成功生成包含 ${Object.keys(confirmedClipsData).length} 个音频的压缩包。`, 'success');

        } catch (error) {
            console.error('打包时出错:', error);
            updateStatus(`打包失败: ${error.message}`, 'error');
        } finally {
            packageBtn.disabled = false;
        }
    }

    // --- 手动加载逻辑 ---
    function handleManualLoad() {
        const videoFile = manualVideoInput.files[0];
        const jsonFile = manualJsonInput.files[0];
        if (!videoFile || !jsonFile) {
            updateStatus('请同时选择视频文件和JSON文件。', 'error');
            return;
        }
        const videoUrl = URL.createObjectURL(videoFile);
        const jsonReader = new FileReader();
        jsonReader.onload = (event) => {
            try {
                const jsonData = JSON.parse(event.target.result);
                loadDataAndSetupEditor(videoUrl, jsonData);
            } catch (e) {
                updateStatus(`手动加载JSON失败: ${e.message}`, 'error');
            }
        };
        jsonReader.readAsText(jsonFile);
    }

    // --- 工具函数 ---
    function updateClipStatusIndicator(clip) {
        const indicator = document.getElementById(`status-${clip.id}`);
        if (!indicator) return;
        switch (clip.status) {
            case 'confirmed':
                indicator.textContent = '✔ 已确认';
                indicator.className = 'sentence-status confirmed';
                break;
            case 'ignored':
                indicator.textContent = '❌ 已忽略';
                indicator.className = 'sentence-status ignored';
                break;
            default:
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

    function updatePackageButton() {
        const count = Object.keys(confirmedClipsData).length;
        packageBtn.textContent = `📦 生成并下载 ZIP (${count} 个文件)`;
        packageBtn.disabled = count === 0;
    }

    function updateStatus(message, type = 'info') {
        statusDisplay.textContent = message;
        statusDisplay.className = 'status-info'; // Reset
        if (type === 'success') statusDisplay.style.backgroundColor = '#c6f6d5';
        else if (type === 'error') statusDisplay.style.backgroundColor = '#fed7d7';
    }

    // --- 启动程序 ---
    init();
});
