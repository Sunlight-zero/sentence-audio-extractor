document.addEventListener('DOMContentLoaded', function () {
    // 初始化 WaveSurfer
    const wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 'rgb(200, 0, 200)',
        progressColor: 'rgb(100, 0, 100)',
        
        // -- 新增：启用横向滚动 --
        // 每秒音频渲染至少 50 像素，对于长音频会超出容器宽度
        minPxPerSec: 50, 
        
        // -- 新增：播放时自动滚动视图 --
        autoScroll: true, // 确保光标在视图内
        autoCenter: true, // 尽量让光标保持在中间

        // -- 新增：启用时间轴插件 --
        plugins: [
            WaveSurfer.Timeline.create({
                container: '#wave-timeline', // 指定时间轴的容器
                // 可以自定义时间轴的样式
                height: 20,
                timeInterval: 1,
                primaryLabelInterval: 5,
                secondaryLabelInterval: 1,
                style: {
                    fontSize: '12px',
                    color: '#666',
                },
            }),
        ],
    });

    // 播放/暂停按钮
    const playBtn = document.getElementById('playBtn');
    playBtn.addEventListener('click', () => {
        wavesurfer.playPause();
    });

    // 监听播放状态并更新按钮文本
    wavesurfer.on('play', () => {
        playBtn.textContent = '暂停';
    });
    wavesurfer.on('pause', () => {
        playBtn.textContent = '播放';
    });
    wavesurfer.on('finish', () => {
        playBtn.textContent = '播放'; // 播放结束后重置按钮
    });

    // 处理用户选择的文件
    const fileInput = document.getElementById('audioFile');
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // 在加载新文件前，确保按钮是“播放”状态
            wavesurfer.pause();
            playBtn.textContent = '播放';
            // 使用用户选择的文件加载波形
            wavesurfer.load(URL.createObjectURL(file));
        }
    });

    // 当波形图准备好后，可以做一些事情，比如打印时长
    wavesurfer.on('ready', (duration) => {
        console.log(`音频加载完成，总时长: ${duration.toFixed(2)} 秒`);
    });
});