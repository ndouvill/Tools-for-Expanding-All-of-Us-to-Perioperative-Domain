from collections import namedtuple
from enum import Enum
import re

import pandas as pd

from .allofus_tool import AllofUsTool


class CodeType(Enum):
    CPT4, ICD10PCS = range(2)


ProcedureCode = namedtuple('ProcedureCode', ['type', 'code'])
CategoryResult = namedtuple('CategoryResult', ['code_type', 'code', 'is_cardiac', 'cardiac_subclass', 'procedural_category'])


class SurgicalProcedureClassifier(AllofUsTool):
    ICD10_CATEGORY_FILE = 'data/icd10pcs_category.csv'
    ICD10_CARDIAC_CLASSIFICATION_FILE = 'data/icd10pcs_cardiac_classification.csv'
    CCS_LABEL_FILE = 'data/ccs_label.csv'
    CPT_CCS_MAPPING_FILE = 'data/cpt_ccs_mapping.csv'
    COVERED_CODE_TYPES = [ct.name for ct in CodeType]
    CARDIAC_SUBCLASS_CODE_TO_NAME = {1: 'Open', 2: 'EP/Cath', 3: 'Transcatheter/Endovascular', 4: 'Other'}

    def __init__(self, workspace_cdr):
        super().__init__(workspace_cdr)

        df_icd10 = pd.read_csv(self.ICD10_CATEGORY_FILE, header=0)
        self.icd10_code_to_category = {row.Code: row.Category for row in df_icd10.itertuples()}

        df_icd10_cardiac_classification = pd.read_csv(self.ICD10_CARDIAC_CLASSIFICATION_FILE,
                                                      header=0,
                                                      dtype={'cardiac_category_code': int})
        self.cardiac_icd10_code_to_category_code = {row.icd10_code: row.cardiac_category_code
                                                    for row in df_icd10_cardiac_classification.itertuples()}

        df_ccs_label = pd.read_csv(self.CCS_LABEL_FILE, header=0, dtype={'ccs_numerical_label': int})
        self.ccs_num_label_to_text = {row.ccs_numerical_label: row.ccs_label for row in df_ccs_label.itertuples()}

        df_cpt_ccs_mapping = pd.read_csv(self.CPT_CCS_MAPPING_FILE, header=0, dtype={'ccs_numerical_label': int})
        self.cpt_to_ccs = {}

        for row in df_cpt_ccs_mapping.itertuples():
            code_from, code_to = row.cpt_code_range.split('-')
            code_start_alpha = ''
            code_end_alpha = ''
            code_num_start_idx = 0
            code_num_end_idx = len(code_from)
            
            if re.search(r'^[A-Z]', code_from):
                code_start_alpha = code_from[0]
                code_num_start_idx = 1
                
            if re.search(r'[A-Z]$', code_from):
                code_end_alpha = code_from[-1]
                code_num_end_idx = len(code_from) - 1
                
            code_from_num = int(code_from[code_num_start_idx:code_num_end_idx])
            code_to_num = int(code_to[code_num_start_idx:code_num_end_idx])
            code_num_len = code_num_end_idx - code_num_start_idx
            
            for code_num in range(code_from_num, code_to_num + 1):
                self.cpt_to_ccs[f'{code_start_alpha}{code_num:0{code_num_len}d}{code_end_alpha}'] = row.ccs_numerical_label

    def classify_surgical_procedure(self, proc_id):
        query_proc_code = f"""
            SELECT
                concept.vocabulary_id,
                concept.concept_code
            FROM
                {self.dataset}.procedure_occurrence
                JOIN {self.dataset}.concept
                    ON procedure_occurrence.procedure_concept_id = concept.concept_id
            WHERE
                procedure_occurrence.procedure_occurrence_id = {proc_id}
        """
        df_proc_code = pd.read_gbq(query_proc_code, dialect='standard')

        if len(df_proc_code) == 0:
            return None

        code_type = df_proc_code.iloc[0]['vocabulary_id']
        code = df_proc_code.iloc[0]['concept_code']

        if code_type not in self.COVERED_CODE_TYPES:
            return None

        proc_code = ProcedureCode(CodeType[code_type], code)
        is_cardiac = self.is_cardiac_procedure(proc_code)

        if is_cardiac:
            if proc_code.type == CodeType.CPT4:
                cardiac_subclass_code = self.cardiac_procedure_type(proc_code.code)
            elif proc_code.type == CodeType.ICD10PCS:
                cardiac_subclass_code = self.cardiac_icd10_code_to_category_code.get(proc_code.code, 4)

            cardiac_subclass = self.CARDIAC_SUBCLASS_CODE_TO_NAME[cardiac_subclass_code]
        else:
            cardiac_subclass = None

        if proc_code.type == CodeType.CPT4:
            procedural_category = self.ccs_num_label_to_text.get(self.cpt_to_ccs.get(proc_code.code, None), 'Does not map to any procedure in range')
        elif proc_code.type == CodeType.ICD10PCS:
            procedural_category = self.icd10_code_to_category[proc_code.code]
        else:
            procedural_category = None

        return CategoryResult(code_type, code, is_cardiac, cardiac_subclass, procedural_category)

    def is_cardiac_procedure(self, concept_code):
        if concept_code.type == CodeType.ICD10PCS:
            if self.icd10_code_to_category[concept_code.code] in ('CARD', 'CBGB', 'CBGC', 'HTP', 'PACE'):
                return True
            else:
                return False
        elif concept_code.type == CodeType.CPT4:
            if self.cardiac_procedure_type(concept_code.code) > 0:
                return True
            else:
                return False

        return None

    def cardiac_procedure_type(self, cpt_code):
        if self.is_open_cardiac_surgical_cpt(cpt_code):
            return 1

        if self.is_ep_cath_surgical_cpt(cpt_code):
            return 2

        if self.is_transcatheter_endovascular_surgical_cpt(cpt_code):
            return 3

        if self.is_other_cardiac_surgical_cpt(cpt_code):
            return 4

        return 0

    def is_obstetric_anesthesia(self, cpt_code):
        if cpt_code in ('01961', '01963', '01967', '01968', '01969'):
            return True
        
        return False

    def is_open_cardiac_surgical_cpt(self, cpt_code):
        if not cpt_code.isdigit():
            return False

        cpt_code = int(cpt_code)

        if (((33020 <= cpt_code <= 33100) and (cpt_code != '33025'))
            or (33120 <= cpt_code <= 33130)
            or (33140 <= cpt_code <= 33141)
            or (33300 <= cpt_code <= 33315)
            or (33321 <= cpt_code <= 33322)
            or (cpt_code == 33335)
            or (33390 <= cpt_code <= 33417)
            or (33422 <= cpt_code <= 33471)
            or (33474 <= cpt_code <= 33476)
            or (cpt_code == 33478)
            or (cpt_code == 33496)
            or (33500 <= cpt_code <= 33507)
            or (cpt_code == 33508)
            or (33510 <= cpt_code <= 33516)
            or (33517 <= cpt_code <= 33530)
            or (33533 <= cpt_code <= 33548)
            or (cpt_code == 33572)
            or (33600 <= cpt_code <= 33622)
            or (33641 <= cpt_code <= 33697)
            or (33702 <= cpt_code <= 33722)
            or (33724 <= cpt_code <= 33732)
            or (33735 <= cpt_code <= 33768)
            or (33770 <= cpt_code <= 33783)
            or (33786 <= cpt_code <= 33788)
            or (33800 <= cpt_code <= 33853)
            or (33858 <= cpt_code <= 33877)
            or (33910 <= cpt_code <= 33926)
            or (33927 <= cpt_code <= 33945)
            or (33975 <= cpt_code <= 33983)):
            return True
        
        return False

    def is_open_cardiac_anesthesia_cpt(self, cpt_code):
        if cpt_code in ('00561', '00562', '00563', '00566', '00567', '00580'):
            return True
        
        return False

    def is_ep_cath_surgical_cpt(self, cpt_code):
        if not cpt_code.isdigit():
            return False

        cpt_code = int(cpt_code)
        
        if ((33016 <= cpt_code <= 33019)
            or (33202 <= cpt_code <= 33275)
            or (33285 <= cpt_code <= 33286)
            or (cpt_code == 33289)
            or (92920 <= cpt_code <= 92979)
            or (92950 <= cpt_code <= 92985)
            or (cpt_code == 92998)
            or (93451 <= cpt_code <= 93533)
            or (93600 <= cpt_code <= 93662)):
            return True
        
        return False

    def is_transcatheter_endovascular_surgical_cpt(self, cpt_code):
        if not cpt_code.isdigit():
            return False

        cpt_code = int(cpt_code)
        
        if ((cpt_code == 33340)
            or (33361 <= cpt_code <= 33364)
            or (33418 <= cpt_code <= 33420)
            or (cpt_code == 33477)
            or (33880 <= cpt_code <= 33891)
            or (33990 <= cpt_code <= 33993)
            or (cpt_code == 92986)
            or (cpt_code == 92987)
            or (cpt_code == 92990)
            or (93580 <= cpt_code <= 93592)):
            return True
        
        return False

    def is_other_cardiac_surgical_cpt(self, cpt_code):
        if not cpt_code.isdigit():
            return False

        cpt_code = int(cpt_code)
        
        if ((33016 <= cpt_code <= 33019)
            or (cpt_code == 33025)
            or (cpt_code == 35820)
            or (cpt_code == 35840)
            or (33365 <= cpt_code <= 33369)
            or (cpt_code == 33320)
            or (cpt_code == 33330)
            or (33946 <= cpt_code <= 33959)
            or (33962 <= cpt_code <= 33974)
            or (33984 <= cpt_code <= 33989)
            or (cpt_code == 33999)):
            return True
        
        return False
