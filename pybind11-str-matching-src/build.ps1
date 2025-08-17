conda activate workflow
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
cp build\Release\fast_match.cp310-win_amd64.pyd ..\waveform-webpage\utils\fast_match.cp310-win_amd64.pyd