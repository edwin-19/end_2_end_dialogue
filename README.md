# End-to-End Japanese LLM Generation  

This repository demonstrates a **Japanese LLM-based audio Q&A** system.  
The key components used are **Ultravox** and **KokoroTTS** for text generation and speech synthesis.  

## 🎙️ Sample Question and Answer Audio in Japanese  

Below is an example of a **generated question** in Japanese:  

### 🎤 Question Audio  
**Question:** 安保繊維が紙巻きに用いられている理由は何ですか?  

<audio controls>
  <source src="question/5.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>  

### 🎧 Answer Audio  
And here is the **corresponding generated answer**:  

**Answer:** それは、**「アンボーン」**という**化学物質**の**作用**と**関連**があります。  
**アンボーン**は、**化学物質**として**作用**する**仕組み**を理解する必要があります。  

<audio controls>
  <source src="output/5.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>  

This project dynamically generates **Japanese audio Q&A** based on text input.  

---

## 🚀 Installation  
First, install the required dependencies:  
```bash
pip install -r requirements.txt
```

## Running the code

### Mock JP Translated

1️⃣ Generate Japanese Text-Based Q&A

This is a pipeline created using the normal ultarvox which is english but i translated it to japanese and they synthesized
```bash
python demo.py
```

### Pure Japanese QnA
2️⃣ Generate Japanese Audio Questions

This pipeline generates the code in japanese but we need japanese audio so i generated this from the same dataset
```bash
python generate_question.py
```

3️⃣ Run the Full Japanese Audio Q&A Pipeline

Run QA pipeline Audio in pure japanese
```
python demo_jp.py 
```

Hardware
- 1x 3090 RTX
- 1x Xeon Intel Processor
- 48 GB RAM
- 500 GB Storage SSD

ToDO in the future:
- [ ] Streaming component
- [ ] Deployment server
- [ ] Finetuning pipeline