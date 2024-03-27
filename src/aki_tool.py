from collections import namedtuple
from enum import Enum
import re

import pandas as pd


class CodeType(Enum):
    CPT4, ICD10PCS = range(2)


ProcedureCode = namedtuple('ProcedureCode', ['type', 'code'])
CategoryResult = namedtuple('CategoryResult', ['code_type', 'code', 'is_cardiac', 'cardiac_subclass', 'procedural_category'])
PostopCreatinine = namedtuple('PostopCreatinine', ['within_2_days', 'within_7_days'])


class AKITool:
    ICD10_CATEGORY_FILE = 'icd10pcs_category.csv'
    ICD10_CARDIAC_CLASSIFICATION_FILE = 'icd10pcs_cardiac_classification.csv'
    CCS_LABEL_FILE = 'ccs_label.csv'
    CPT_CCS_MAPPING_FILE = 'cpt_ccs_mapping.csv'
    COVERED_CODE_TYPES = [ct.name for ct in CodeType]
    SURGICAL_PROC_CONCEPT_ID = 4301351
    SERUM_CREATININE_CONCEPT_IDS = {3016723, 3020564}
    UNIT_SOURCE_VALUES_FOR_SERUM_CREATININE = {'mg/dL', '258797006', 'umol/L'}
    MALE_CONCEPT_ID = 45880669
    CARDIAC_SUBCLASS_CODE_TO_NAME = {1: 'Open', 2: 'EP/Cath', 3: 'Transcatheter/Endovascular', 4: 'Other'}

    def __init__(self, workspace_cdr):
        self.dataset = workspace_cdr

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

        query_anes_cpt_concepts = f"""
            SELECT
                concept_id,
                concept_code
            FROM
                {self.dataset}.concept
            WHERE
                vocabulary_id = 'CPT4'
                AND REGEXP_CONTAINS(concept_code, r'^(0[01][0-9]{{3}})$')
        """
        df_anes_cpt_concepts = pd.read_gbq(query_anes_cpt_concepts, dialect='standard')
        self.anes_cpt_concept_ids = set(df_anes_cpt_concepts['concept_id'])

        query_surgical_concept_ids = f"""
            WITH
                target_concept AS (
                    SELECT DISTINCT procedure_concept_id
                    FROM {self.dataset}.procedure_occurrence
                )
            SELECT
                descendant_concept_id AS surgical_concept_id
            FROM
                {self.dataset}.concept_ancestor
                JOIN {self.dataset}.concept
                    ON descendant_concept_id = concept_id
                        AND ancestor_concept_id <> descendant_concept_id
            WHERE
                vocabulary_id = 'CPT4'
                AND standard_concept = 'S'
                AND ancestor_concept_id = {self.SURGICAL_PROC_CONCEPT_ID}
                AND domain_id IN (
                    SELECT domain_ID
                    FROM {self.dataset}.concept
                    WHERE concept_id = {self.SURGICAL_PROC_CONCEPT_ID}
                )
                AND descendant_concept_id IN (
                    SELECT *
                    FROM target_concept
                )
            UNION DISTINCT
            SELECT
                procedure_concept_id AS surgical_concept_id
            FROM
                target_concept
                JOIN {self.dataset}.concept
                    ON procedure_concept_id = concept_id
            WHERE
                vocabulary_id = 'ICD10PCS'
                AND concept_code IN ({','.join(map(lambda x: f"'{x}'", df_icd10['Code']))})
            UNION DISTINCT
            SELECT
                concept_id AS surgical_concept_id
            FROM
                {self.dataset}.concept
            WHERE
                concept_id = {self.SURGICAL_PROC_CONCEPT_ID}
                AND vocabulary_id = 'CPT4'
                AND standard_concept = 'S'
        """
        df_surgical_concept_ids = pd.read_gbq(query_surgical_concept_ids, dialect='standard')
        self.all_surgical_concept_ids = set(df_surgical_concept_ids['surgical_concept_id'])

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

    def acute_kidney_injury(self, proc_id):
        highest_preop_creatinine = self.get_highest_preop_creatinine(proc_id)

        if highest_preop_creatinine is None:
            return -3
        
        preop_egfr_lowest = self.get_preop_egfr(proc_id)

        if preop_egfr_lowest is not None and preop_egfr_lowest < 15:
            return -2
        
        postop_creatinine = self.get_postop_creatinine(proc_id)

        if postop_creatinine.within_7_days is None:
            return -1

        has_close_surgery = self.has_close_surgery_without_creatinine_between_surgeries(proc_id)

        if has_close_surgery:
            return -4

        baseline_creatinine = highest_preop_creatinine

        if postop_creatinine.within_7_days >= 3.0 * baseline_creatinine:
            return 3
        elif postop_creatinine.within_7_days >= 2.0 * baseline_creatinine:
            return 2
        elif postop_creatinine.within_7_days >= 1.5 * baseline_creatinine:
            return 1
        elif (postop_creatinine.within_2_days is not None
              and postop_creatinine.within_2_days >= 0.3 + baseline_creatinine):
            return 1

        return 0

    def get_highest_preop_creatinine(self, proc_id):
        query_highest_preop_creatinine = f"""
            SELECT
                procedure_occurrence.procedure_occurrence_id,
                measurement.value_as_number,
                measurement.unit_source_value
            FROM
                {self.dataset}.procedure_occurrence
                JOIN {self.dataset}.measurement
                    ON procedure_occurrence.person_id = measurement.person_id
            WHERE
                procedure_occurrence.procedure_occurrence_id = {proc_id}
                AND measurement_concept_id IN ({','.join(map(str, self.SERUM_CREATININE_CONCEPT_IDS))})
                AND unit_source_value IN (
                    {','.join(map(lambda x: f"'{x}'", self.UNIT_SOURCE_VALUES_FOR_SERUM_CREATININE))}
                )
                AND value_as_number IS NOT NULL
                AND measurement_datetime
                    BETWEEN DATETIME_SUB(procedure_occurrence.procedure_datetime, INTERVAL 60 DAY)
                        AND procedure_occurrence.procedure_datetime
        """
        df_highest_preop_creatinine = pd.read_gbq(query_highest_preop_creatinine, dialect='standard')

        if len(df_highest_preop_creatinine) == 0:
            return None

        max_creatinine = -1.0

        for row in df_highest_preop_creatinine.itertuples():
            creatinine = row.value_as_number
            unit = row.unit_source_value

            if unit == 'umol/L':
                creatinine /= 88.4
            elif unit not in ['mg/dL', '258797006']:
                continue

            if creatinine < 0.2 or creatinine >= 25:
                continue

            max_creatinine = max(max_creatinine, creatinine)

        if max_creatinine < 0.0:
            return None
        
        return max_creatinine

    def get_preop_egfr(self, proc_id):
        query_preop_egfr = f"""
            WITH
                body_height AS (
                    SELECT
                        procedure_occurrence_id,
                        body_height,
                        body_height_unit
                    FROM
                        (
                            SELECT
                                procedure_occurrence.procedure_occurrence_id,
                                measurement.measurement_datetime,
                                measurement.value_as_number AS body_height,
                                measurement.unit_source_value AS body_height_unit,
                                ROW_NUMBER() OVER (ORDER BY measurement.measurement_datetime DESC) AS row_num
                            FROM
                                {self.dataset}.procedure_occurrence
                                JOIN {self.dataset}.measurement
                                    ON procedure_occurrence.visit_occurrence_id = measurement.visit_occurrence_id
                            WHERE
                                procedure_occurrence.procedure_occurrence_id = {proc_id}
                                AND measurement.measurement_concept_id = 3036277
                                AND measurement.value_as_number IS NOT NULL
                        )
                    WHERE
                        row_num = 1
                )
            SELECT
                measurement.measurement_datetime,
                measurement.value_as_number,
                measurement.unit_source_value,
                FLOOR(
                    SAFE_CAST(FORMAT_DATE('%Y.%m%d%H%M%S', measurement.measurement_datetime) AS FLOAT64)
                    - SAFE_CAST(FORMAT_DATE('%Y.%m%d%H%M%S', person.birth_datetime) AS FLOAT64)
                ) AS age,
                person.gender_concept_id,
                body_height.body_height
            FROM
                {self.dataset}.procedure_occurrence
                JOIN {self.dataset}.measurement
                    ON procedure_occurrence.person_id = measurement.person_id
                JOIN {self.dataset}.person
                    ON measurement.person_id = person.person_id
                LEFT JOIN body_height
                    ON procedure_occurrence.procedure_occurrence_id = body_height.procedure_occurrence_id
            WHERE
                procedure_occurrence.procedure_occurrence_id = {proc_id}
                AND measurement.measurement_concept_id IN ({','.join(map(str, self.SERUM_CREATININE_CONCEPT_IDS))})
                AND measurement.unit_source_value IN (
                    {','.join(map(lambda x: f"'{x}'", self.UNIT_SOURCE_VALUES_FOR_SERUM_CREATININE))}
                )
                AND measurement.value_as_number IS NOT NULL
                AND measurement.measurement_datetime
                    BETWEEN DATETIME_SUB(procedure_occurrence.procedure_datetime, INTERVAL 60 DAY)
                        AND procedure_occurrence.procedure_datetime
        """
        df_preop_egfr = pd.read_gbq(query_preop_egfr, dialect='standard')
        max_creatinine = -1.0

        for row in df_preop_egfr.itertuples():
            creatinine = row.value_as_number
            unit = row.unit_source_value

            if unit == 'umol/L':
                creatinine /= 88.4
            elif unit not in ['mg/dL', '258797006']:
                continue

            if creatinine < 0.2 or creatinine >= 25:
                continue
                
            max_creatinine = max(max_creatinine, creatinine)
            
        if max_creatinine < 0:
            return None

        age = df_preop_egfr['age'].iloc[0]

        if age >= 18:
            is_male = True if df_preop_egfr['gender_concept_id'].iloc[0] == self.MALE_CONCEPT_ID else False

            inner_factor = max_creatinine / (0.9 if is_male else 0.7)
            alpha_exponent = -0.302 if is_male else -0.241
            final_multiplier = 1.0 if is_male else 1.01

            egfr = (142
                    * pow(min(inner_factor, 1.0), alpha_exponent)
                    * pow(max(inner_factor, 1.0), -1.2)
                    * pow(0.9938, age)
                    * final_multiplier)
        else:
            height = df_preop_egfr['body_height'].iloc[0]
            
            if pd.isna(height):
                return None
                
            egfr = 0.413 * height / max_creatinine
            
        if egfr <= 0 or egfr >= 300:
            return None
            
        return round(egfr)

    def get_postop_creatinine(self, proc_id):
        query_postop_creatinine = f"""
            SELECT
                measurement.measurement_datetime,
                measurement.value_as_number,
                measurement.unit_source_value
            FROM
                {self.dataset}.procedure_occurrence
                JOIN {self.dataset}.measurement
                    ON procedure_occurrence.person_id = measurement.person_id
            WHERE
                procedure_occurrence.procedure_occurrence_id = {proc_id}
                AND measurement.measurement_concept_id IN ({','.join(map(str, self.SERUM_CREATININE_CONCEPT_IDS))})
                AND measurement.unit_source_value IN (
                    {','.join(map(lambda x: f"'{x}'", self.UNIT_SOURCE_VALUES_FOR_SERUM_CREATININE))}
                )
                AND measurement.value_as_number IS NOT NULL
                AND measurement.measurement_datetime > procedure_occurrence.procedure_datetime
                AND measurement.measurement_datetime <= DATETIME_ADD(procedure_occurrence.procedure_datetime, INTERVAL {{within_n_days}} DAY)
        """
        postop_creatinine = []

        for within_n_days in (2, 7):
            df_postop_creatinine = pd.read_gbq(query_postop_creatinine.format(within_n_days=within_n_days), dialect='standard')
            max_creatinine = -1.0

            for row in df_postop_creatinine.itertuples():
                creatinine = row.value_as_number
                unit = row.unit_source_value

                if unit == 'umol/L':
                    creatinine /= 88.4
                elif unit not in ['mg/dL', '258797006']:
                    continue

                if creatinine < 0.2 or creatinine >= 25:
                    continue
                    
                max_creatinine = max(max_creatinine, creatinine)

            if max_creatinine < 0.0:
                max_creatinine = None
                
            postop_creatinine.append(max_creatinine)

        return PostopCreatinine(*postop_creatinine)

    def has_close_surgery_without_creatinine_between_surgeries(self, proc_id):
        query_serum_creatinine_datetime = f"""
            SELECT
                measurement.measurement_datetime,
            FROM
                {self.dataset}.procedure_occurrence
                JOIN {self.dataset}.measurement
                    ON procedure_occurrence.person_id = measurement.person_id
            WHERE
                procedure_occurrence.procedure_occurrence_id = {proc_id}
                AND measurement.measurement_concept_id IN ({','.join(map(str, self.SERUM_CREATININE_CONCEPT_IDS))})
                AND measurement.unit_source_value IN (
                    {','.join(map(lambda x: f"'{x}'", self.UNIT_SOURCE_VALUES_FOR_SERUM_CREATININE))}
                )
                AND measurement.value_as_number IS NOT NULL
                AND measurement.measurement_datetime IS NOT NULL
            ORDER BY
                measurement_datetime
        """
        df_serum_creatinine_datetime = pd.read_gbq(query_serum_creatinine_datetime, dialect='standard')
        creatinine_datetimes = list(df_serum_creatinine_datetime['measurement_datetime'])

        query_person_id = f"""
            SELECT
                person_id
            FROM
                {self.dataset}.procedure_occurrence
            WHERE
                procedure_occurrence_id = {proc_id}
        """
        df_person_id = pd.read_gbq(query_person_id, dialect='standard')
        person_id = df_person_id.iloc[0, 0]

        query_procedures = f"""
            WITH
                creatinine_meas_visit AS (
                    SELECT DISTINCT
                        visit_occurrence_id
                    FROM
                        {self.dataset}.measurement
                    WHERE
                        person_id = {person_id}
                        AND value_as_number IS NOT NULL
                        AND measurement_concept_id IN ({','.join(map(str, self.SERUM_CREATININE_CONCEPT_IDS))})
                        AND visit_occurrence_id IS NOT NULL
                )
            SELECT
                procedure_occurrence.procedure_occurrence_id,
                procedure_occurrence.procedure_datetime
            FROM
                creatinine_meas_visit
                JOIN {self.dataset}.procedure_occurrence
                    ON creatinine_meas_visit.visit_occurrence_id = procedure_occurrence.visit_occurrence_id
                JOIN {self.dataset}.concept AS proc_concept
                    ON procedure_occurrence.procedure_concept_id = proc_concept.concept_id
                JOIN {self.dataset}.person
                    ON procedure_occurrence.person_id = person.person_id
                JOIN {self.dataset}.concept AS gender_concept
                    ON person.gender_concept_id = gender_concept.concept_id
                JOIN {self.dataset}.concept AS race_concept
                    ON person.race_concept_id = race_concept.concept_id
            WHERE
                procedure_occurrence.person_id = {person_id}
                AND procedure_occurrence.procedure_concept_id IN ({','.join(map(str, self.all_surgical_concept_ids))})
                AND FLOOR(SAFE_CAST(FORMAT_DATE('%Y.%m%d%H%M%S', procedure_occurrence.procedure_datetime) AS FLOAT64)
                        - SAFE_CAST(FORMAT_DATE('%Y.%m%d%H%M%S', person.birth_datetime) AS FLOAT64)) >= 18
        """
        df_procedures = pd.read_gbq(query_procedures, dialect='standard')
        sorted_unique_proc_datetimes = sorted(set(df_procedures['procedure_datetime']))
        proc_dt_to_next_proc_dt = {sorted_unique_proc_datetimes[i]: sorted_unique_proc_datetimes[i + 1]
                                   for i in range(len(sorted_unique_proc_datetimes) - 1)}

        for row in df_procedures.itertuples():
            if row.procedure_occurrence_id != proc_id:
                continue

            proc_datetime = row.procedure_datetime

            if proc_datetime not in proc_dt_to_next_proc_dt:
                return False
            
            next_proc_datetime = proc_dt_to_next_proc_dt[proc_datetime]

            if next_proc_datetime - proc_datetime > pd.Timedelta(7, 'day'):
                return False

            if len(set(filter(lambda x: x > proc_datetime, creatinine_datetimes))
                   & set(filter(lambda x: x <= next_proc_datetime, creatinine_datetimes))) == 0:
                return True

        return False

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
