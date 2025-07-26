// static/script.js

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'
// 新增：导入 Timeline 插件
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'

document.addEventListener('DOMContentLoaded', function () {
    const API_BASE_URL = 'http://127.0.0.1:5000';

    // 获取 HTML 元素
    const videoFileInput = document.getElementById('videoFile');
    const sentenceInput = document.getElementById('targetSentence');
    const startTaskBtn = document.getElementById('startTaskBtn');
    const statusDisplay = document.getElementById('statusDisplay');
    const correctorUI = document.getElementById('corrector-ui');
    const videoElement = document.getElementById('videoPlayer');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const playSelectionBtn = document.getElementById('play-selection-btn');
    const saveBtn = document.getElementById('save-btn');

    let wavesurfer = null;
    let activeRegion = null;
    let pollingInterval = null;

    startTaskBtn.addEventListener('click', async () => {
        const videoFile = videoFileInput.files[0];
        const sentence = sentenceInput.value.trim();

        if (!videoFile || !sentence) {
            alert('请选择一个视频文件并填写目标句子！');
            return;
        }

        correctorUI.classList.add('hidden');
        if (pollingInterval) clearInterval(pollingInterval);
        statusDisplay.textContent = '正在上传文件并启动任务...';
        statusDisplay.style.backgroundColor = '#e2e8f0';
        startTaskBtn.disabled = true;

        const formData = new FormData();
        formData.append('videoFile', videoFile);
        formData.append('sentence', sentence);

        try {
            const response = await fetch(`${API_BASE_URL}/upload_and_start_task`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '启动任务失败');
            
            statusDisplay.textContent = '后端已接收任务，处理中... (这可能需要几分钟)';
            startPolling();
        } catch (error) {
            statusDisplay.textContent = `错误: ${error.message}`;
            statusDisplay.style.backgroundColor = '#fed7d7';
            startTaskBtn.disabled = false;
        }
    });

    function startPolling() {
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/get_task_result`);
                const data = await response.json();

                if (data.status === 'pending_correction') {
                    clearInterval(pollingInterval);
                    statusDisplay.textContent = 'AI 处理完成，请开始校对！';
                    statusDisplay.style.backgroundColor = '#c6f6d5';
                    startTaskBtn.disabled = false;
                    setupCorrectorUI(data);
                } else if (data.status === 'error') {
                    clearInterval(pollingInterval);
                    statusDisplay.textContent = `处理失败: ${data.error}`;
                    statusDisplay.style.backgroundColor = '#fed7d7';
                    startTaskBtn.disabled = false;
                } else {
                    statusDisplay.textContent += '.';
                }
            } catch (error) {
                clearInterval(pollingInterval);
                statusDisplay.textContent = `轮询错误: ${error.message}`;
                statusDisplay.style.backgroundColor = '#fed7d7';
                startTaskBtn.disabled = false;
            }
        }, 3000);
    }

    function setupCorrectorUI(data) {
        correctorUI.classList.remove('hidden');
        videoElement.src = API_BASE_URL + data.video_url;

        if (wavesurfer) {
            wavesurfer.destroy();
        }

        // 关键修改：初始化插件
        // 1. Regions 插件
        const regions = RegionsPlugin.create();

        // 2. Timeline 插件
        const timeline = TimelinePlugin.create({
            container: '#timeline',
            // 自定义时间格式，例如 M:SS
            formatTimeCallback: (seconds) => {
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = (seconds % 60).toFixed(0);
                return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
            },
        });

        // 关键修改：重新配置 WaveSurfer 实例
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgb(135, 168, 206)',
            progressColor: 'rgb(43, 83, 131)',
            media: videoElement,
            // 新增：播放时自动滚动波形图
            autoScroll: true,
            // 新增：为长音频设置一个最小的像素/秒（即缩放级别），防止波形图过长
            minPxPerSec: 100,
            // 关键修改：使用 `plugins` 数组注册所有插件
            plugins: [regions, timeline],
        });

        wavesurfer.on('decode', () => {
            regions.clearRegions();
            activeRegion = regions.addRegion({
                start: data.initial_start,
                end: data.initial_end,
                color: 'rgba(246, 173, 85, 0.25)',
                drag: true,
                resize: true,
            });
            updateTimestampInputs();
        });

        regions.on('region-updated', (region) => {
            activeRegion = region;
            updateTimestampInputs();
        });

        // --- 关键修正: 监听 region-out 事件以暂停播放 ---
        let isPlayingRegion = false;
        regions.on('region-out', (region) => {
            // 当播放光标离开区域时，如果是由“播放选区”功能启动的，则暂停
            if (isPlayingRegion) {
                isPlayingRegion = false;
                wavesurfer.pause();
            }
        });

        // 当播放结束时（例如到达视频末尾），也重置标志
        wavesurfer.on('finish', () => {
            isPlayingRegion = false;
        });
        
        // 修改“播放选区”按钮的逻辑
        playSelectionBtn.onclick = () => {
            if (activeRegion) {
                isPlayingRegion = true;
                activeRegion.play();
            }
        };
    }

    function updateTimestampInputs() {
        if (activeRegion) {
            startTimeInput.value = activeRegion.start.toFixed(3);
            endTimeInput.value = activeRegion.end.toFixed(3);
        }
    }

    saveBtn.addEventListener('click', async () => {
        if (!activeRegion) {
            alert('没有可保存的选区。');
            return;
        }
        const finalStartTime = parseFloat(startTimeInput.value);
        const finalEndTime = parseFloat(endTimeInput.value);

        statusDisplay.textContent = '正在保存校对结果...';
        statusDisplay.style.backgroundColor = '#e2e8f0';

        try {
            const response = await fetch(`${API_BASE_URL}/save_correction`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start_time: finalStartTime, end_time: finalEndTime }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '保存失败');
            statusDisplay.textContent = `任务完成！已保存到: ${data.output_path}`;
            statusDisplay.style.backgroundColor = '#c6f6d5';
            correctorUI.classList.add('hidden');
        } catch (error) {
            statusDisplay.textContent = `保存错误: ${error.message}`;
            statusDisplay.style.backgroundColor = '#fed7d7';
        }
    });
});
