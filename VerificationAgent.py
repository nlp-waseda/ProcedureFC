import json
import asyncio
from typing import List
import regex

from LLMAgent import LLMAgent
from LLMModel import LLMModel
from GoogleSE import GoogleSE

class VerificationAgent(LLMAgent):
    
    def __init__(self, llm_model: LLMModel, prompt_base: List[str], max_attempt_num: int):
        """prompt_base: [statement, flowchart]"""
        super().__init__(llm_model, prompt_base, max_attempt_num)

    async def verify(self, evidence: str, statement: str, max_evidence_length: int = 30000) -> dict:

        """
        returns a dictionary with keys 'reason' and 'result'.
        or returns a string if there is a notation error
        """

        input = self._prompt_base[0].replace(
            '<input-evidence>', evidence[:max_evidence_length]
        ).replace(
            '<input-statement>', statement
        )

        response_dict = {}
        finished = False
        for _ in range(self._max_attempt_num):

            print(f"==== {_+1}th attempt for statement verification ==== ") # debug

            response = await self._llm_model.request(input)

            matches = regex.findall(r"<json>(.*?)</json>", response, regex.DOTALL, overlapped=True) # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # responseをjsonで読み込み、dictに変換する
                    if isinstance(response_dict, dict) and isinstance(response_dict.get("reason"), str) and isinstance(response_dict.get("result"), bool):
                        finished = True
                        break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                    else:
                        pass
                except json.JSONDecodeError:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("Verification failed after maximum attempts.")
        
        return response_dict
    
    async def verifyFC(self, evidence: str, flowchart: str, max_evidence_length: int = 30000) -> dict:

        """
        returns a dictionary with keys 'reason' and 'result'.
        """

        input = self._prompt_base[1].replace(
            '<input-evidence>', evidence[:max_evidence_length]
        ).replace(
            '<input-flowchart>', flowchart
        )

        response_dict = {}
        finished = False
        for _ in range(self._max_attempt_num):

            print(f"==== {_+1}th attempt for flowchart verification ==== ") # debug

            response = await self._llm_model.request(input)

            matches = regex.findall(r"<json>(.*?)</json>", response, regex.DOTALL, overlapped=True) # <json>と</json>の間を抜き出す（リーズニングの過程タグが出力され、マッチ範囲がネストした場合、再帰的に検索を行い最後のマッチ結果を取り出す）
            if len(matches) > 0:
                json_text = matches[-1].strip()
                try:
                    response_dict = json.loads(json_text) # responseをjsonで読み込み、dictに変換する
                    if isinstance(response_dict, dict) and isinstance(response_dict.get("reason"), str) and isinstance(response_dict.get("result"), bool):
                        finished = True
                        break # 全てのフォーマットチェックをクリアしたので、試行ループを抜ける
                    else:
                        pass
                except json.JSONDecodeError:
                    pass
            else:
                pass

        if finished == False:
            raise RuntimeError("Flowchart Verification failed after maximum attempts.")
        
        return response_dict

# module test

if __name__ == "__main__":

    from FlowchartAgent import FlowchartAgent
    from DecompositionAgent import DecompositionAgent
    from QueryGenerator import QueryGenerator
    from vLLMModel import vLLMModel
    from OpenAIAPI import OpenAIAPI

    async def main():

        #llm_model = OpenAIAPI("gpt-5.1", {"temperature": 0.5})
        """
        llm_model = vLLMModel('Qwen/Qwen3-30B-A3B-Instruct-2507', {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 3000,
            "min_p": 0
        })
        """
        #"""
        llm_model = vLLMModel('openai/gpt-oss-120b', {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": -1,
            "max_tokens": 10000
        })
        #"""

        prompt_base = []
        with open('prompts/prompt_verify1.2.6.txt', encoding='utf-8') as f:
            prompt_base.append(f.read())
        with open('prompts/prompt_verifyFC0.0.1.txt', encoding='utf-8') as f:
            prompt_base.append(f.read())
        verification_agent = VerificationAgent(llm_model, prompt_base, 5)

        flowchart = "%% Procedure to obtain and cook \"Fine Tea, Full Moon\" in Genshin Impact\nflowchart TD\n    A[\"Check Event Availability (-> &quot;A Tea Party Beneath the Moon&quot; event in Genshin Impact)\"] --> B[\"Participate in the Event Quests\"]\n    B --> C[\"Unlock the Recipe\"]\n    C -->|\"recipe for &quot;Fine Tea, Full Moon&quot;\"| D[\"Learn the Recipe\"]\n    D --> E[\"Gather Ingredients: Tea Leaves, Sugar, Lotus Head, Milk\"]\n    E -->|\"ingredients: Tea Leaves, Sugar, Lotus Head, Milk\"| F[\"Cook the Dish: Fine Tea, Full Moon\"]"
        statement = "In Genshin Impact, the dish called \"Fine Tea, Full Moon\" is not part of the permanent recipe pool and is typically only available during certain seasonal events in Liyue, often around the in-game Mid-Autumn Festival period."
        evidence = "Fine Tea, Full Moon | Genshin Impact Wiki | Fandom\nSign In\nRegister\nGenshin Impact Wiki\nExplore\nMain Page\nDiscuss\nAll Pages\nCommunity\nInteractive Maps\nRecent Blog Posts\nGuidelines\nCommunity Rules\nEditing Guidelines\nGeneral Guidelines\nSyntax Guidelines\nPage Layout Guides\nSection Layout Guides\nOther Languages Guide\nImages Policy\nVideo Guide Policy\nCaution Against Visual Editor\nEditor Dashboard\nWiki Staff\nCharacters\nPlayable Characters\nTraveler\nAmber\nLisa\nKaeya\nCharacter List\nFeatured Characters\nNefer\nWonderland Manekin\nFurina\nArlecchino\nZhongli\nCharacter Comparison\nNPCs\nNotable NPCs\nFactions\nThe Seven\nAdventurers' Guild\nKnights of Favonius\nLiyue Qixing\nInazuma Shogunate\nSumeru Akademiya\nPalais Mermonia\nSpeaker's Chamber\nFatui\nVoice Actors\nThe World\nTeyvat\nMondstadt\nLiyue\nInazuma\nSumeru\nFontaine\nNatlan\nNod-Krai\nSnezhnaya\nKhaenri'ah\nLore\nTimeline\nLanguages\nBooks\nSoundtrack\nManga\nComics\nQuests\nArchon Quests\nStory Quests\nEvent Quests\nWorld Quests\nCommissions\nHangout Events\nTribal Chronicles\nEvents\nVersion\nIn-Game Events\nLogin Events\nFlagship Events\nWeb Events\nEvent History\nExploration\nStatue of The Seven\nOculi\nShrines of Depths\nTeleport Waypoints\nMap\nViewpoints\nWildlife\nCombat\nAttributes\nElements\nElemental Reactions\nEnemies\nCommon Enemies\nElite Enemies\nNormal Bosses\nWeekly Bosses\nTalents\nConstellations\nActivities\nMiliastra Wonderland\nDomains\nLey Line Outcrops\nExpeditions\nFishing\nImaginarium Theater\nSpiral Abyss\nStygian Onslaught\nRepertoire of Myriad Melodies\nShops\nOther\nItems\nWeapons\nBows\nCatalysts\nClaymores\nPolearms\nSwords\nWeapon Enhancement Materials\nArtifacts\nArtifact Sets\nArtifact EXP\nCharacter Development Items\nCharacter EXP Materials\nCharacter Ascension Materials\nCharacter Level-Up Materials\nCharacter and Weapon Enhancement Materials\nRefinement Materials\nCharacter Talent Materials\nWeapon Ascension Materials\nMaterials\nLocal Specialties\nCooking Ingredients\nForging Materials\nFurnishing Materials\nGardening Materials\nConsumables\nFood\nPotions\nGadgets\nOther\nOriginal Resin\nNamecards\nBundles\nCurrencies\nQuest Items\nEvent Items\nPrecious Items\nFurnishings\nProgression\nAdventure Rank\nCharacter Experience\nAdventurer Handbook\nAchievements\nReputation\nOfferings\nSerenitea Pot\nGenius Invokation TCG\nMonetization\nWishes\nPrimogems\nGenesis Crystal\nBattle Pass\nBlessing of the Welkin Moon\nCharacter Outfits\nGift Shop\nPaimon's Bargains\nCrafting\nAlchemy\nCreation\nCooking\nForging\nProcessing\nOther Systems\nInventory\nArchive\nTutorials\nAccount\nCo-Op Mode\nControls\nFriends\nProfile\nSettings\nCommunity\nGenshin Impact\nmiHoYo\nHOYO-MiX\nHoYoLAB\nGlossary\nSign In\nDon't have an account?\nRegister\nSign In\nMenu\nExplore\nMore\nHistory\nAdvertisement\nSkip to content\nGenshin Impact Wiki\n34,347pages\nExplore\nMain Page\nDiscuss\nAll Pages\nCommunity\nInteractive Maps\nRecent Blog Posts\nGuidelines\nCommunity Rules\nEditing Guidelines\nGeneral Guidelines\nSyntax Guidelines\nPage Layout Guides\nSection Layout Guides\nOther Languages Guide\nImages Policy\nVideo Guide Policy\nCaution Against Visual Editor\nEditor Dashboard\nWiki Staff\nCharacters\nPlayable Characters\nTraveler\nAmber\nLisa\nKaeya\nCharacter List\nFeatured Characters\nNefer\nWonderland Manekin\nFurina\nArlecchino\nZhongli\nCharacter Comparison\nNPCs\nNotable NPCs\nFactions\nThe Seven\nAdventurers' Guild\nKnights of Favonius\nLiyue Qixing\nInazuma Shogunate\nSumeru Akademiya\nPalais Mermonia\nSpeaker's Chamber\nFatui\nVoice Actors\nThe World\nTeyvat\nMondstadt\nLiyue\nInazuma\nSumeru\nFontaine\nNatlan\nNod-Krai\nSnezhnaya\nKhaenri'ah\nLore\nTimeline\nLanguages\nBooks\nSoundtrack\nManga\nComics\nQuests\nArchon Quests\nStory Quests\nEvent Quests\nWorld Quests\nCommissions\nHangout Events\nTribal Chronicles\nEvents\nVersion\nIn-Game Events\nLogin Events\nFlagship Events\nWeb Events\nEvent History\nExploration\nStatue of The Seven\nOculi\nShrines of Depths\nTeleport Waypoints\nMap\nViewpoints\nWildlife\nCombat\nAttributes\nElements\nElemental Reactions\nEnemies\nCommon Enemies\nElite Enemies\nNormal Bosses\nWeekly Bosses\nTalents\nConstellations\nActivities\nMiliastra Wonderland\nDomains\nLey Line Outcrops\nExpeditions\nFishing\nImaginarium Theater\nSpiral Abyss\nStygian Onslaught\nRepertoire of Myriad Melodies\nShops\nOther\nItems\nWeapons\nBows\nCatalysts\nClaymores\nPolearms\nSwords\nWeapon Enhancement Materials\nArtifacts\nArtifact Sets\nArtifact EXP\nCharacter Development Items\nCharacter EXP Materials\nCharacter Ascension Materials\nCharacter Level-Up Materials\nCharacter and Weapon Enhancement Materials\nRefinement Materials\nCharacter Talent Materials\nWeapon Ascension Materials\nMaterials\nLocal Specialties\nCooking Ingredients\nForging Materials\nFurnishing Materials\nGardening Materials\nConsumables\nFood\nPotions\nGadgets\nOther\nOriginal Resin\nNamecards\nBundles\nCurrencies\nQuest Items\nEvent Items\nPrecious Items\nFurnishings\nProgression\nAdventure Rank\nCharacter Experience\nAdventurer Handbook\nAchievements\nReputation\nOfferings\nSerenitea Pot\nGenius Invokation TCG\nMonetization\nWishes\nPrimogems\nGenesis Crystal\nBattle Pass\nBlessing of the Welkin Moon\nCharacter Outfits\nGift Shop\nPaimon's Bargains\nCrafting\nAlchemy\nCreation\nCooking\nForging\nProcessing\nOther Systems\nInventory\nArchive\nTutorials\nAccount\nCo-Op Mode\nControls\nFriends\nProfile\nSettings\nCommunity\nGenshin Impact\nmiHoYo\nHOYO-MiX\nHoYoLAB\nGlossary\nin:\nFood, 3-Star Food, Food with Recipes,\nand\n9 more\nRecovery Dishes\nHealing Dishes\nLiyue Dishes\nHP Restore Dishes\nHP Restore Percent Dishes\nHP Restore Fixed Dishes\nCooking\nShop Availability\nReleased in Version 4.4\nEnglish\nEspañol\nFrançais\n日本語\nPortuguês do Brasil\nРусский\nTiếng Việt\n中文\nFine Tea, Full Moon\nSign in to edit\nHistory\nPurge\nTalk (0)\nFine Tea, Full Moon\nNormal\nDelicious\nSuspicious\nDescrip­tion\nA tea pastry containing egg yolk. The salted yolk is first steamed, then stuffed and wrapped in tea powder-infused dough. It's then kneaded into the shape of a full moon before being baked. This results in a crispy, flaky exterior with a salty and sweet interior, the different layers forming a delicious and addictive delicacy.\nEffect\nRestores 32% of Max HP and an additional 1,250 HP to the selected character.\nDescription\nA tea pastry containing egg yolk. The loose and delicate texture of the grainy egg yolk mingles with a faint fragrance of tea. One small mouthful brings earthy happiness enough to make you swoon. You could pop several in your mouth at once, and woe betide any who should seek to steal a single crumb from the corner of your lips.\nEffect\nRestores 34% of Max HP and an additional 1,900 HP to the selected character.\nDescription\nA tea pastry containing egg yolk. As you pick it up, layers flake away like loose earth in a landslide. The first bite is sticky, sweet, cloying... And uh-oh, something gets stuck in your throat. Get some water to wash it down, then think about what to do with the rest.\nEffect\nRestores 30% of Max HP and an additional 600 HP to the selected character.\nType\nRecovery Dishes\nQuality\nProficiency\n15\nRegion\nLiyue\nHow to Obtain\nEffects\nRecipe\nComplete Qiaoying of the Sacred Mountain\nSource 1\nSold by Lianfang\nHP RestoreHP Restore PercentHP Restore Fixed\nFine Tea, Full Moon is a food item that the player can cook. The recipe for Fine Tea, Full Moon is obtained during World Quest Qiaoying of the Sacred Mountain in Series Chenyu's Blessings of Sunken Jade.\nContents\n1 Recipe\n1.1 Manual Cooking\n2 Shop Availability\n3 Trivia\n4 Other Languages\n5 Change History\n6 Navigation\nRecipe[]\nCooking 3 Chenyu Adeptea 2 Bird Egg 2 Flour 1 Sugar 1 Fine Tea, Full Moon\nManual Cooking[]\nShop Availability[]\nThere is 1 Shop that sells Fine Tea, Full Moon:\nItem\nNPC\nMora Cost\nStock\nNotes\nFine Tea, Full Moon\nLianfang\n6,000\n2\nDaily\nTrivia[]\nNo Mail has Fine Tea, Full Moon, Delicious Fine Tea, Full Moon, or Suspicious Fine Tea, Full Moon as an attachment.\nOther Languages[]\nLanguageOfficial NameLiteral MeaningEnglishFine Tea, Full Moon—Chinese(Simplified)茶好月圆Chá Hǎo Yuè YuánGood Tea Full MoonChinese(Traditional)茶好月圓Chá Hǎo Yuè YuánJapanese良茶満月Ryoucha Mangetsu‍‍[!]Assumed reading‍Good Tea Full MoonKorean차와 보름달Chawa BoreumdalTea and Full MoonSpanishPastel de té y lunaTea and Moon CakeFrenchPleine lune au thé exquisFull Moon with Exquisite TeaRussian«Вкусный чай, полная луна»\"Vkusnyy chay, polnaya luna\"\"Tasty Tea, Full Moon\"ThaiFine Tea, Full Moon—VietnameseTrà Hảo Nguyệt ViênGermanMundender Tee, voller MondTasty Tea, Full MoonIndonesianFine Tea, Full Moon—PortugueseBolinho Lua de CháTurkishÇaylı DolunayItalianLune piene con tèFull Moons with Tea\nChange History[]\nReleased in Version 4.4[Create New History]\nNavigation[]\nNormal Dishes\nCategories\nCategories:\nFood\n3-Star Food\nFood with Recipes\nRecovery Dishes\nHealing Dishes\nLiyue Dishes\nHP Restore Dishes\nHP Restore Percent Dishes\nHP Restore Fixed Dishes\nCooking\nShop Availability\nReleased in Version 4.4\nLanguages\nEspañol\nFrançais\n日本語\nPortuguês do Brasil\nРусский\nTiếng Việt\n中文\nCommunity content is available under CC-BY-SA unless otherwise noted.\nMore Fandoms\nFantasy\nGenshin Impact\nAdvertisement\nExplore properties\nFandom\nFanatical\nGameSpot\nMetacritic\nTV Guide\nHonest Entertainment\nFollow Us\nOverview\nWhat is Fandom?\nAbout\nCareers\nPress\nContact\nTerms of Use\nPrivacy Policy\nDigital Services Act\nGlobal Sitemap\nLocal Sitemap\nCommunity\nCommunity Central\nSupport\nHelp\nAdvertise\nMedia Kit\nContact\nFandom Apps\nTake your favorite fandoms with you and never miss a beat.\nGenshin Impact Wiki is a Fandom Games Community.\nView Mobile Site"
        verify_fc_result = await verification_agent.verifyFC(evidence, flowchart, max_evidence_length=576000)
        print("Verify FC Result:\n", verify_fc_result)
        verify_statement_result = await verification_agent.verify(evidence, statement, max_evidence_length=576000)
        print("Verify Statement Result:\n", verify_statement_result)

    asyncio.run(main())