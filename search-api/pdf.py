from PyPDF2 import PdfMerger

merger = PdfMerger()
files = [
    r"C:\Users\AOT\Downloads\get_payslip  Jan 22  2026.pdf",
    r"C:\Users\AOT\Downloads\get_payslip Jan8 2026.pdf",
    r"C:\Users\AOT\Downloads\get_payslip Dec24  2025.pdf",
    r"C:\Users\AOT\Downloads\get_payslip  Dec 12 2025.pdf",
]

for pdf in files:
    merger.append(pdf)

merger.write(r"C:\Users\AOT\Downloads\Sreeja_Suresh-Latest_PaySlips_Merged.pdf")
merger.close()
