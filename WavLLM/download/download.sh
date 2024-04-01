# WavLLM model
url_p1="https://valle.blob.core.windows.net/share/wavllm/fi"
url_p2="nal.pt?sv=2021-10-04&st=2024-04-01T02%3A27%3A"
url_p3="42Z&se=2025-11-27T02%3A27%3A00Z&sr=b&sp=r&si"
url_p4="g=g52Mdaf9bQajzrhQcKk2WWWy5vmGBTqP1vRJ1zelCoE%3D"
curl -o final.pt ${url_p1}${url_p2}${url_p3}${url_p4}

# gaokao_audio
url_p1="https://valle.blob.core.windows.net/share/wavllm/ga"
url_p2="okao_audio.zip?sv=2021-10-04&st=2024-04-01T02%3A"
url_p3="35%3A41Z&se=2025-11-27T02%3A35%3A00Z&sr=b&sp=r&s"
url_p4="ig=FWwBtnylTbPV1WS0apaGxvsykwQm3G4stg4%2Bhi%2BnvbY%3D"
curl -o gaokao_audio.zip ${url_p1}${url_p2}${url_p3}${url_p4}

# gaokao_transcript
url_p1="https://valle.blob.core.windows.net/share/wavllm/ga"
url_p2="okao_text.zip?sv=2021-10-04&st=2024-04-01T02%3A40%3A"
url_p3="25Z&se=2025-11-27T02%3A40%3A00Z&sr=b&sp=r&s"
url_p4="ig=Su%2FzZGMhhPZ9mmG7i%2BcNEiPqFsuqkdLRiLHyX2WH%2BVM%3D"
curl -o gaokao_text.zip ${url_p1}${url_p2}${url_p3}${url_p4}
