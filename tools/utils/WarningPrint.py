class WarningPrint():
    def __init__(self):
        self.warning_data = []

    def getWarning(self, person_id, action, frame_index):
        if person_id < len(self.warning_data):
            if frame_index % 15 == 0:
                if action == 'nature':
                    self.warning_data[person_id].text = ''
                else:
                    self.warning_data[person_id].text = action
        else:
            if frame_index % 15 == 0:
                objTmp = WarningObj(action)
                self.warning_data.append(objTmp)
            else:
                objTmp = WarningObj('')
                self.warning_data.append(objTmp)
        return self.warning_data[person_id].text


class WarningObj():
    def __init__(self, text):
        self.text = text