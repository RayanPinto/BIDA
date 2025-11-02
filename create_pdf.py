"""
Script to create a PDF with all simplified programs
Uses only built-in Python libraries
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

def create_pdf():
    """Create PDF with all simplified programs"""
    
    # Create PDF document
    doc = SimpleDocTemplate("Simplified_Lab_Programs.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkgreen
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.darkred
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=8,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=6,
        fontName='Courier'
    )
    
    # Build content
    story = []
    
    # Title page
    story.append(Paragraph("Simplified Lab Programs for Exam", title_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("This document contains simplified versions of all lab programs that use only built-in Python libraries. These programs are designed to be easy to understand and implement during exams without internet access.", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Programs Included:", heading_style))
    story.append(Paragraph("1. EXP4 - Association Rules Mining (Apriori Algorithm)", styles['Normal']))
    story.append(Paragraph("2. EXP5 - Sentiment Analysis using Naive Bayes", styles['Normal']))
    story.append(Paragraph("3. EXP6 - Web Scraping and Topic Modeling", styles['Normal']))
    story.append(Paragraph("4. EXP7 - HTML Sentiment Analysis", styles['Normal']))
    story.append(Paragraph("5. EXP8 - LDA Topic Modeling", styles['Normal']))
    
    story.append(PageBreak())
    
    # Read and add each program
    programs = [
        ("EXP4_Simplified.py", "EXP4 - Association Rules Mining (Apriori Algorithm)"),
        ("EXP5_Simplified.py", "EXP5 - Sentiment Analysis using Naive Bayes"),
        ("EXP6_Simplified.py", "EXP6 - Web Scraping and Topic Modeling"),
        ("EXP7_Simplified.py", "EXP7 - HTML Sentiment Analysis"),
        ("EXP8_Simplified.py", "EXP8 - LDA Topic Modeling")
    ]
    
    for filename, title in programs:
        if os.path.exists(filename):
            story.append(Paragraph(title, heading_style))
            story.append(Spacer(1, 10))
            
            # Read the Python file
            with open(filename, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Add code as preformatted text
            story.append(Preformatted(code_content, code_style))
            story.append(PageBreak())
        else:
            story.append(Paragraph(f"{title} - File not found: {filename}", subheading_style))
            story.append(Spacer(1, 10))
    
    # Add documentation
    story.append(Paragraph("Documentation", heading_style))
    story.append(Spacer(1, 10))
    
    if os.path.exists("Simplified_Programs_Documentation.md"):
        with open("Simplified_Programs_Documentation.md", 'r', encoding='utf-8') as f:
            doc_content = f.read()
        
        # Split into sections and add as paragraphs
        sections = doc_content.split('\n## ')
        for i, section in enumerate(sections):
            if i == 0:
                # First section (no ## prefix)
                lines = section.split('\n')
                for line in lines:
                    if line.strip():
                        if line.startswith('#'):
                            story.append(Paragraph(line[1:].strip(), heading_style))
                        else:
                            story.append(Paragraph(line, styles['Normal']))
            else:
                # Other sections (add back ## prefix)
                lines = ('## ' + section).split('\n')
                for line in lines:
                    if line.strip():
                        if line.startswith('###'):
                            story.append(Paragraph(line[3:].strip(), subheading_style))
                        elif line.startswith('##'):
                            story.append(Paragraph(line[2:].strip(), heading_style))
                        else:
                            story.append(Paragraph(line, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print("PDF created successfully: Simplified_Lab_Programs.pdf")

if __name__ == "__main__":
    create_pdf()
