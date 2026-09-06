import pytest
from text_preprocessing.text_preprocessing import clean_cv_text

def test_preserve_technical_skills():
    input_text = "Skills: C++, C#, C, R, .NET, Node.js, React.js, Docker, CI/CD, AWS, Go, Vue.js, F#, ASP.NET"
    cleaned = clean_cv_text(input_text)
    
    expected_skills = [
        "c++", "c#", "c", "r", ".net", "node.js", 
        "react.js", "docker", "ci/cd", "aws", "go", "vue.js", "f#", "asp.net"
    ]
    
    for skill in expected_skills:
        assert skill in cleaned.split(), f"Failed to preserve skill: {skill}"

if __name__ == "__main__":
    test_preserve_technical_skills()
    print("Test passed: Technical skills preserved.")
