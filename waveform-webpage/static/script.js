document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    
    const statusContainer = document.getElementById('status-container');
    const statusMessage = document.getElementById('status-message');
    const spinner = document.getElementById('spinner');

    const resultContainer = document.getElementById('result-container');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const proofreadLinkBtn = document.getElementById('proofread-link-btn');

    let pollInterval;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 禁用按钮并显示加载状态
        submitBtn.disabled = true;
        submitBtn.textContent = '正在上传...';
        resultContainer.style.display = 'none';
        statusContainer.style.display = 'block';
        spinner.style.display = 'block';
        statusMessage.textContent = '文件上传中，请稍候...';

        const formData = new FormData(form);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '上传失败，服务器返回错误。');
            }

            const data = await response.json();
            
            if (data.status === 'processing' && data.task_id) {
                submitBtn.textContent = '处理中...';
                statusMessage.textContent = '文件上传成功，后台正在处理...';
                // 开始轮询任务状态
                pollStatus(data.task_id);
            } else {
                throw new Error('未能获取有效的任务ID。');
            }

        } catch (error) {
            console.error('处理请求时出错:', error);
            statusMessage.textContent = `错误: ${error.message}`;
            spinner.style.display = 'none';
            submitBtn.disabled = false;
            submitBtn.textContent = '开始处理';
        }
    });

    function pollStatus(taskId) {
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/task_status/${taskId}`);
                if (!res.ok) {
                    throw new Error('无法连接服务器检查状态。');
                }
                const data = await res.json();

                statusMessage.textContent = data.message || data.error || `当前状态: ${data.status}`;

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    spinner.style.display = 'none';
                    statusContainer.style.display = 'none';
                    resultContainer.style.display = 'block';
                    
                    // 设置下载和跳转链接
                    downloadJsonBtn.href = data.result_json_url;
                    downloadJsonBtn.download = data.result_json_url.split('/').pop();

                    const proofreadUrl = `/proofread?video_url=${encodeURIComponent(data.video_url)}&json_url=${encodeURIComponent(data.result_json_url)}`;
                    proofreadLinkBtn.href = proofreadUrl;

                    // 恢复按钮状态
                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始处理';

                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    spinner.style.display = 'none';
                    statusMessage.textContent = `处理失败: ${data.error}`;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '重新处理';
                }
            } catch (error) {
                console.error('轮询状态时出错:', error);
                statusMessage.textContent = `轮询错误: ${error.message}`;
                clearInterval(pollInterval);
                spinner.style.display = 'none';
                submitBtn.disabled = false;
                submitBtn.textContent = '重新处理';
            }
        }, 3000); // 每3秒查询一次状态
    }
});
