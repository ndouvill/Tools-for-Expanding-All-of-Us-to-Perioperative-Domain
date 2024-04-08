from collections import namedtuple

import pandas as pd

from .allofus_tool import AllofUsTool


PostopCreatinine = namedtuple('PostopCreatinine', ['within_2_days', 'within_7_days'])


class AKITool(AllofUsTool):
    ICD10_CATEGORY_FILE = 'data/icd10pcs_category.csv'
    SURGICAL_PROC_CONCEPT_ID = 4301351
    SERUM_CREATININE_CONCEPT_IDS = {3016723, 3020564}
    UNIT_SOURCE_VALUES_FOR_SERUM_CREATININE = {'mg/dL', '258797006', 'umol/L'}
    MALE_CONCEPT_ID = 45880669

    def __init__(self, workspace_cdr):
        super().__init__(workspace_cdr)

        df_icd10 = pd.read_csv(self.ICD10_CATEGORY_FILE, header=0)
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
