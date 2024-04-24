stage=$1
# WavLLM model
if [ "$stage" -eq 0 ]; then
  url_p1="https://valle.blob.core.windows.net/share/wavllm/fi"
  url_p2="nal.pt?sv=2021-10-04&st=2024-04-24T04%3A50%3A"
  url_p3="15Z&se=2025-04-25T04%3A50%3A00Z&sr=b&sp=r&si"
  url_p4="g=M82edjKinydPiVd86oS78ZS9L"
  url_p5="TVxg0%2F2om3IaEkodIo%3D"
  curl -o final.pt ${url_p1}${url_p2}${url_p3}${url_p4}${url_p5}
else
  # gaokao_audio
  url_p1="https://valle.blob.core.windows.net/share/wavllm/ga"
  url_p2="okao_audio.zip?sv=2021-10-04&st=2024-04-24T04%3A58%3A"
  url_p3="56Z&se=2025-04-25T04%3A58%3A00Z&sr=b&sp=r&s"
  url_p4="ig=0ql1dkz59%2FSxRHkz1ajtC"
  url_p5="yfCR5Hva4UISlIfDrOO%2BRc%3D"
  curl -o gaokao_audio.zip ${url_p1}${url_p2}${url_p3}${url_p4}${url_p5}

  # gaokao_transcript
  url_p1="https://valle.blob.core.windows.net/share/wavllm/ga"
  url_p2="okao_text.zip?sv=2021-10-04&st=2024-04-24T04%3A57%3A"
  url_p3="37Z&se=2025-04-25T04%3A57%3A00Z&sr=b&sp=r&s"
  url_p4="ig=n5QKXU3F9RiP6SxHl6uVEJ"
  url_p5="8m7WZ3iEeOGns1BoIozvI%3D"
  curl -o gaokao_text.zip ${url_p1}${url_p2}${url_p3}${url_p4}${url_p5}
fi