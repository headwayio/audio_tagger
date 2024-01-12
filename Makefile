all: icd9_codelist.txt

icd9_codelist.zip:
	curl https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip -o icd9_codelist.zip

icd9_codelist.txt: icd9_codelist.zip
	unzip -j icd9_codelist.zip CMS32_DESC_LONG_DX.txt
	mv CMS32_DESC_LONG_DX.txt icd9_codelist.txt
	touch icd9_codelist.txt
	rm icd9_codelist.zip
