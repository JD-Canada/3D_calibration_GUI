document_1_text = 'view1_x'
document_2_text = 'view1_y'

document_1_words = document_1_text.split("_")
document_2_words = document_2_text.split("_")

common = set(document_1_words).intersection( set(document_2_words) )
unique = set(document_1_words).symmetric_difference( set(document_2_words) )

print(common)
