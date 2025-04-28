from resume_parser import ResumeParser
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_my_resume.py path/to/your/resume.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Initialize the parser
    parser = ResumeParser()
    
    print(f"Parsing resume: {pdf_path}")
    result = parser.parse_resume(pdf_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        # Print basic information
        print("\n=== BASIC INFORMATION ===")
        if result["parsed"]["name"]:
            print(f"Name: {result['parsed']['name']}")
        if result["parsed"]["email"]:
            print(f"Email: {result['parsed']['email']}")
        if result["parsed"]["phone"]:
            print(f"Phone: {result['parsed']['phone']}")
        
        # Print number of entries found
        print(f"\nSkills found: {len(result['parsed']['skills'])}")
        print(f"Education entries: {len(result['parsed']['education'])}")
        print(f"Experience entries: {len(result['parsed']['experience'])}")
        
        # Save detailed result to JSON
        output_file = "parsed_resume.json"
        with open(output_file, "w") as f:
            json.dump(result["parsed"], f, indent=2)
        
        print(f"\nComplete parsed resume saved to {output_file}")

if __name__ == "__main__":
    main()