# Psychological Defense Mechanism Coding Handbook

*   **Framework Reference:** This handbook is based on the hierarchical theory of defense mechanisms developed by J. Christopher Perry and operationalized in the Defense Mechanisms Rating Scales (DMRS).

# Introduction

## 1. Research Background and Goal

Psychological defense mechanisms, a core concept from psychoanalytic theory, are unconscious processes that individuals employ to reduce anxiety and maintain psychological equilibrium when facing internal conflicts or external stressors. These mechanisms profoundly influence an individual's emotions, cognitions, and behaviors, holding significant implications for mental health, interpersonal relationships, and the therapeutic process.

The goal of this project is to construct a high-quality emotional support dialogue dataset annotated with defense mechanisms. This will address a critical gap in the field of Natural Language Processing (NLP) and lay the foundation for the application of Artificial Intelligence in mental health.

## 2. Importance of the Annotation Task

As an annotator, you are the most critical part of this project. Your professional, detailed, and consistent annotations are the bedrock of this research. High-quality annotated data will directly determine the performance ceiling and practical value of future AI models.

## 3. Purpose and Structure of the Handbook

This handbook provides a unified, clear, and actionable set of standards and procedures to ensure the highest possible consistency and accuracy in annotation. It covers core concepts, a detailed annotation workflow, definitions and criteria for each label, and guidelines for handling common issues.

# Part 1: Core Concepts and Overview

## 1. A Brief Introduction to Defense Mechanisms

The primary function of defense mechanisms is to protect the psyche from overwhelming emotions such as anxiety, shame, or grief. Based on their adaptiveness, they are organized into a hierarchy, ranging from immature (e.g., reality-distorting denial) and neurotic (e.g., repression) to mature (e.g., constructive humor). This research focuses on identifying the linguistic expression of these mechanisms in text-based dialogues.

## 2. Label System Overview

<table>
  <thead>
    <tr>
        <th>Label</th>
        <th>English Name</th>
        <th>Brief Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>0</td>
        <td>No Defense / Neutral<br/>Utterance</td>
        <td>Purely functional<br/>utterances that serve to<br/>maintain conversational<br/>flow, express social<br/>niceties, or exchange<br/>non-emotional information.</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Action Defenses</td>
        <td>Expresses or discharges<br/>emotions through concrete<br/>actions (e.g., passive<br/>aggression, help-rejecting<br/>complaining, acting out).</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Major Image-Distorting<br/>Defenses</td>
        <td>Distorts the image of self<br/>or others in an extreme,<br/>all-or-nothing manner (e.g.,<br/>splitting, projective<br/>identification).</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Disavowal Defenses</td>
        <td>Evades unpleasant facts or<br/>internal pain through<br/>mechanisms like denial,<br/>rationalization, and<br/>projection.</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Minor Image-Distorting<br/>Defenses</td>
        <td>Distorts the image of self<br/>or others in a less severe<br/>manner (e.g., devaluation,<br/>idealization, omnipotence).</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Neurotic Defenses</td>
        <td>Manages anxiety or internal<br/>conflict at an unconscious<br/>level through repression,<br/>displacement, dissociation,<br/>or reaction formation.</td>
    </tr>
    <tr>
        <td>6</td>
        <td>Obsessional Defenses</td>
        <td>Copes with anxiety by<br/>exerting excessive control<br/>over thoughts or feelings<br/>(e.g., intellectualization,</td>
    </tr>
  </tbody>
</table>

<table>
  <tbody>
    <tr>
        <td> </td>
        <td> </td>
        <td>isolation of affect,<br/>undoing).</td>
    </tr>
    <tr>
        <td>7</td>
        <td>High-Adaptive Defenses</td>
        <td>Modulates emotions in a<br/>mature and constructive<br/>way (e.g., humor,<br/>sublimation, altruism,<br/>self-observation).</td>
    </tr>
    <tr>
        <td>8</td>
        <td>Unclear / Needs More<br/>Information</td>
        <td>Used when information is<br/>insufficient, the utterance is<br/>ambiguous, or it cannot be<br/>clearly classified.</td>
    </tr>
  </tbody>
</table>

### 3. Core Annotation Principles

*   **Primacy of Context:** Annotations must be made by considering the preceding dialogue to understand the true psychological function of an utterance.

*   **Function-Oriented Principle:** Focus on the underlying purpose of the statement. Ask yourself: "What psychological goal is the speaker trying to achieve? Are they avoiding pain or actively coping?"

*   **Distinguishing Emotion from Defense:** The direct expression of an emotion (e.g., "I am very sad") is not classified as a defense mechanism. A defense is only present when an emotional expression is clearly distorted, avoided, or transformed.

*   **Identifying Mature Coping:** Actively look for constructive coping methods that are grounded in an acknowledgment of reality.

## Part 2: Detailed Definitions and Criteria

### Label 0: No Defense / Neutral Utterance

*   **Definition:** This category is reserved for utterances that serve a purely conversational or social function and do not engage with emotional conflict or psychological content. These are statements that are functionally neutral.

*   **Core Function:** To maintain conversational flow, perform social niceties, or exchange logistical information without psychological self-revelation.

*   **Manifestations:**

- Greetings and Farewells: "Hello," "Hi," "Bye," "See you later."

- Simple Social Rituals: "Thank you," "Please," "You're welcome."

- Simple Affirmations/Negations: Short, non-elaborated responses like "Okay," "Yes," "No," or "Right."

### Level 1: Action Defenses

This level involves dealing with stressors by engaging in actions to release tension, often without reflection on negative consequences.

## Passive Aggression

*   **Definition:** Expressing aggression toward others in an indirect and unassertive manner. There is a facade of compliance masking covert resistance or hostility.

*   **Example:** A boss asks an employee to work over the weekend. The employee agrees ("Okay, no problem") but deliberately submits the report late, and it contains many avoidable errors, covertly expressing dissatisfaction.

*   **DMRS-Q Items:**

*   **ITEM 45:** At times when expressing an opinion or wish might be helpful, the subject fails to express himself adequately, instead finding indirect, even annoying ways to show his or her opposition to the influence of others, for example, being silent.

*   **ITEM 88:** The subject fails to stand up for his or her interests and seems to let bad things happen to him or herself that could be prevented, maybe even assuming a ''martyr'' role.

*   **ITEM 89:** While outwardly cooperative or compliant, the individual procrastinates and refuses to do things on time or as asked, even when it would be easy to do so.

*   **ITEM 102:** When angry toward someone significant, the subject takes anger out on himself instead of expressing it directly.

*   **ITEM 116:** The subject has ''a chip on his or her shoulder'' or a grudge, and seems to find reasons to feel unfairly treated, even when he or she is not.

*   **Distinction:** Differentiated from Self-Assertion, which involves expressing thoughts and feelings directly and clearly (e.g., "I have a prior commitment this weekend and cannot work").

*   **Help-Rejecting Complaining**

*   **Definition:** The repetitious use of complaints to ostensibly seek help, while simultaneously rejecting any advice or support that is offered.

*   **Example:** A person complains at length about their problems. When a friend offers a suggestion, they immediately reply, "I've tried that, it didn't work," or "That's a good idea, but..." This leaves the helper feeling frustrated.

*   **DMRS-Q Items:**

*   **ITEM 21:** The subject complains spontaneously about how others don't really care, or have made his or her problems worse, even when there is clear evidence that others have tried to help.

*   **ITEM 84:** The subject recites a litany of issues and problems but does not appear to be engaged in solving them, but rather prefers to complain.

*   **ITEM 127:** The subject tends to exaggerate his or her complaints about a life problem or somatic symptom, making them seem worse or more

significant than they are.
■ **ITEM 130**: The subject complains about life issues or problems as if each were insoluble, and systematically rejects others' suggestions about ways of handling them.
■ **ITEM 149**: When the subject brings up a problem to discuss, others try to address the problem, but in response the subject skips to a different problem, thereby dismissing rather than engaging others in any suggestions offered.
*   **Acting Out**
*   **Definition**: Expressing an unconscious wish or impulse through action to avoid being conscious of the accompanying affect. The action is often impulsive, without regard for negative consequences.
*   **Example**: After a heated argument, a person feels rejected and angry. Instead of processing these emotions, they storm out and engage in reckless driving to release the emotional tension.
*   **DMRS-Q Items**:
■ **ITEM 5**: The subject loses his or her temper easily.
■ **ITEM 76**: In response to interpersonal disappointment or disagreement the subject tends to act impulsively, without reflection or considering the negative consequences.
■ **ITEM 80**: The subject is often inhibited from expressing him or herself, but sometimes acts in uncontrolled ways to get or do something he or she wants, ignoring normal constraints.
■ **ITEM 118**: Whenever the subject feels angry, disappointed or rejected by someone, the subject resorts to uncontrolled behaviors as an escape from distressing feelings, such as binge-eating, drinking, sexual escapades, drug use, reckless driving, or getting into trouble.
■ **ITEM 144**: The subject tends to express feelings, wishes or impulses directly in behavior, not only words, without prior thought. However, afterward, he or she may feel guilty or expect some punishment.

### Level 2: Major Image-Distorting Defenses
This level involves distorting the image of self or others in an extreme, all-or-nothing manner to manage intolerable anxiety.
*   **Splitting**
*   **Definition**: Viewing oneself or others as all-good or all-bad, failing to integrate positive and negative qualities into a cohesive, realistic image.
*   **Example**: A patient initially views her therapist as a perfect savior (idealization). However, after the therapist needs to reschedule once, the patient sees him as selfish and uncaring (devaluation), completely unable to recall any previous positive feelings.

○ **DMRS-Q Items:**
- **Splitting of self-image:**
- **ITEM 3:** The subject has periods of saying highly positive things about him or herself, and other periods saying highly negative things about him or herself, without appearing to notice the contradiction and without addressing it, other than to feel confused about him or herself at moments.
- **ITEM 6:** The subject speaks of him or herself in a wholly negative way at times, as if there is nothing positive or redeeming about him or herself.
- **ITEM 98:** The subject expresses a series of highly unrealistic positive attributes about him or herself whereas at another point the subject sees only negatives in him or herself. The subject dismisses attempts to see things in a balanced more realistic way.
- **ITEM 142:** The subject tends to highlight objects with an emotional meaning that matches his or her own emotional tone at the moment. Any feeling that doesn't match this is ignored or denied.
- **ITEM 145:** Whenever saying something negative about him or herself, the subject rejects others attempts to explore positive or more balanced views, and paradoxically becomes even more confirmed in his or her own worthlessness.
- **Splitting of other's image:**
- **ITEM 35:** The subject experiences other people and objects in "black or white" terms, failing to form more realistic views that balance positive and negative aspects of them.
- **ITEM 61:** The subject attributes unrealistic positive characteristics to an object, such as being all-powerful, omni-benevolent, a savior. Because of the unrealistic belief that the positive object will take care of one's problems, the subject ignores the need to take care of some of his or her own needs.
- **ITEM 92:** The subject attributes unrealistic negative characteristics to an object, such as being all-powerful, malevolent, threatening. As a result, he or she makes some effort to protect him or herself from its influence, even though this response appears unwarranted or exaggerated.
- **ITEM 94:** The subject fails to recognize that someone may be untrustworthy, hurtful, or manipulative and does not draw obvious conclusions based on their behavior. This generally results in using very poor judgment about how others will treat the subject.
- **ITEM 114:** The subject expresses hatred toward someone or

something and refuses to acknowledge anything that does not confirm the hatred.

o **Distinction**: More severe than Devaluation/Idealization. Splitting is an absolute "all-good" or "all-bad" state, whereas devaluation/idealization are "exaggerated" assessments where some reality testing remains.

*   **Projective Identification**

o **Definition**: The individual projects an unacceptable affect onto another person but does not disavow the feeling. Instead, they misattribute their own feeling as a justifiable reaction to the other person and, through their behavior, induce the very feeling in the other that they first mistakenly believed to be there.

o **Example**: A client who is unconsciously angry acts suspiciously toward the therapist. When the therapist asks a neutral question, the client responds, "What do you mean by that? Do you think I'm pathetic?" The client's provocative behavior eventually causes the therapist to feel irritated, at which point the client says, "See! I knew you hated me!"

o **DMRS-Q Items**:

- ■ **ITEM 72**: Sometimes the subject gets angry or fearful toward someone for no apparent reason, but then accuses the other person of intending to make him or her feel that way.

- ■ **ITEM 75**: At times the subject's feelings merge with those of another person and the subject assumes the other's feelings and needs are exactly the same as the subject's own. He or she then tends to "put words in the other's mouth".

- ■ **ITEM 101**: In conversations, the subject sometimes seems confused about distinguishing his or her own feelings from those of the other person.

- ■ **ITEM 103**: When the subject gets upset at someone, he or she gets very angry and loses control, but then blames the other person for making him or her lose control. Nonetheless, the subject may feel some guilt for losing control.

- ■ **ITEM 113**: The subject feels provoked by someone when no obvious provocation is apparent. As the subject becomes angry, accusatory or verbally abusive, the subject provokes the same negative feelings in the other which the subject mistakenly believed the other person had at the outset.

### Level 3: Disavowal Defenses

This level involves refusing to acknowledge unacceptable aspects of internal experience or external reality.

*   **Denial**

o **Definition**: Refusing to acknowledge some aspect of external reality or

personal experience that would be apparent to others.
*   **Example**: A patient diagnosed with a serious illness continues to act as if nothing is wrong, telling family, "I feel fine, the doctors made a mistake."
*   **DMRS-Q Items**:
*   **ITEM 20**: When confronted with topics that might be personally meaningful, the subject denies they are important and refuses to talk about them further.
*   **ITEM 33**: Contrary to the evidence from the interview, the subject claims to have done something that in all likelihood he or she did not do, and may become irritated if confronted with any discrepancy.
*   **ITEM 121**: Whenever talking about potentially distressing events or experiences, the subject strongly claims not to have any feelings about the topic, although this seems highly unlikely.
*   **ITEM 124**: Whenever asked about things the subject did or felt, the subject denies any involvement, does not want to talk about them or avoids explaining his or her reluctance.
*   **ITEM 137**: The subject is hard to talk with, responding to many questions with answers like ''no'' or ''not really'' and does not elaborate, rather than giving some fuller answers which one would normally expect.
*   **Distinction**: Differentiated from Repression, which pushes an internal wish or thought out of awareness, whereas denial rejects an external reality.
*   **Rationalization**
*   **Definition**: Devising reassuring or self-serving but incorrect explanations for one's own or others' behavior.
*   **Example**: An employee criticized for being late says, "It wasn't my fault; the traffic was terrible and my alarm didn't go off."
*   **DMRS-Q Items**:
*   **ITEM 19**: To avoid taking responsibility for one's actions or misdeeds, the subject makes excuses or points out others' contributions to the problem, thereby minimizing his or her own role.
*   **ITEM 42**: The subject avoids feelings of guilt or shame by justifying his actions or by referring to external reasons that impelled him to act.
*   **ITEM 59**: When discussing a problem that the subject contributed to, the subject explains his or her own actions far more than necessary, as if explaining away his or her own fault.
*   **ITEM 86**: Whenever confronted about his or her own feelings or intentions, the subject avoids acknowledging them by giving a plausible explanation that covers up the real subjective reasons.
*   **ITEM 120**: Whenever discussing something uncomfortable about him or herself, the subject tries to convince someone else of a more positive explanation, as if lying to him or herself about the truth.
*   **Distinction**: Differentiated from Intellectualization. Rationalization finds an

excuse for a specific past action to avoid blame. Intellectualization uses abstract, generalized thinking to avoid feeling emotions about a topic.

* **Projection**

* **Definition:** Falsely attributing one's own unacknowledged feelings, impulses, or thoughts to others.

* **Example:** A person who has strong, unacknowledged feelings of hostility toward a coworker constantly feels that the coworker is targeting them.

* **DMRS-Q Items:**

* **ITEM 112:** When others comment or inquire about the subject's own feelings, actions, or intentions, the subject is very elusive or frankly denies the material, but the subject subsequently talks about similar feelings, actions, intentions, etc., in others.

* **ITEM 115:** When experiencing or confronted with a problem, the subject shames, humiliates, or blames someone else for the problem, ignoring his or her own role.

* **ITEM 123:** An attitude of suspiciousness or prejudice toward a group of other individuals, allows the subject not to express an interest in the same motives or feelings but remain blind to them in him or herself.

* **ITEM 134:** When others ask the subject questions, the subject is suspicious about others real reasons or motives for the question.

* **ITEM 141:** The subject perceives others as untrustworthy, unfaithful, or manipulative when there is no objective basis for these concerns. This may even appear paranoid.

* **Distinction:** Differentiated from Projective Identification. In projection, the feeling is disowned entirely ("You are angry, not me"). In projective identification, the individual still experiences the feeling but believes it's a justified reaction to the other person, whom they have provoked.

* **Autistic Fantasy**

* **Definition:** Engaging in excessive daydreaming as a substitute for human relationships, direct action, or problem-solving.

* **Example:** A shy individual who is afraid to ask someone on a date spends hours fantasizing about a romantic relationship with them instead of taking any real action.

* **DMRS-Q Items:**

* **ITEM 2:** The subject has repetitive or serial daydreams to which he or she retreats in lieu of real life social relationships.

* **ITEM 24:** The subject daydreams a lot, not in a way that leads to creative planning or action, but simply for its own gratification, in lieu of action.

* **ITEM 106:** In dealing with some problems, the subject prefers to daydream about solutions, as a substitute for planning direct, realistic, and effective actions.

* **ITEM 110:** Whenever being self-assertive would be helpful, the subject

may act passively but later withdraw into fantasies of being assertive or aggressive toward others as a compensation.

*   **ITEM 148:** The subject gets intensely involved in fantasy roles or actions that express wishes and feelings that the subject does not express in real life. For example, living out a role in a social situation or game or which has no connection to real life ways in which the subject expresses him or herself.

# Level 4: Minor Image-Distorting Defenses

This level involves propping up self-esteem by distorting one's self-image, but in a way that is less encompassing than major image distortion.

*   **Devaluation**

- **Definition:** Attributing exaggeratedly negative qualities to oneself or others.

- **Example:** (Devaluing others) "That new car he bought is just for show; it's a waste of money." (Devaluing self) "I'm a complete idiot for failing that test."

- **DMRS-Q Items:**

- **Devaluation of self-image:**

- **ITEM 12:** The subject says demeaning things about him whether somewhat funny or not-such as "I am so-ooooo stupid."

- **ITEM 29:** The subject makes a lot of unwarranted negative, sarcastic, or biting statements about the self, but the individual can acknowledge some of their positive aspects, if these are pointed out.

- **ITEM 34:** When experiencing failure, disappointment, shame or loss of self-esteem, the subject dismisses the issue by saying something negative about him or herself, then dismisses the problem by moving to another topic and avoids focusing on the feelings.

- **ITEM 56:** The subject is preoccupied with real or exaggerated faults in him or herself, although he or she can acknowledge some realistic positive aspects, if these are pointed out.

- **ITEM 147:** When confronted by a personal disappointment the subject makes negative comments about him or herself but then avoids further discussion of the disappointment in any detail.

- **Devaluation of other's image:**

- **ITEM 54:** When a topic brings with it feelings of disappointment, shame or loss of self-esteem, the subject dismisses the issue by finding some fault or criticism elsewhere or by uttering obscene comments about it.

- **ITEM 82:** The subject devalues others' accomplishments or motives, to minimize their significance, but he or she quickly

dismisses such topics rather than dwell on them.
*   **ITEM 85**: When asked to discuss something about him or herself, the subject diverts the focus to saying negative things about others, as if devaluing others will raise his or her own self-esteem.
*   **ITEM 111**: The subject has negative things to say about a lot of individuals or objects, although he or she can acknowledge some of their positive aspects, if these are pointed out.
*   **ITEM 143**: The subject makes sarcastic or biting statements about others to minimize their positive qualities and dismiss any competition or threat they may pose.
*   **Distinction**: Less severe than Splitting. Devaluation is an "exaggerated negative evaluation," but the person may still acknowledge some positive qualities.
*   **Idealization**
*   **Definition**: Attributing exaggeratedly positive qualities to oneself or others.
*   **Example**: "My therapist is a genius; only he can solve my problems."
*   **DMRS-Q Items**:
*   **Idealization of self-image**:
*   **ITEM 38**: When confronted with any negative aspects of him or herself, the subject appears to downplay or ignore them by substituting talk about positive self-attributes instead.
*   **ITEM 71**: The subject makes many references to how important he or she is with an emphasis on self-image, rather than real accomplishments which might make the person important to others.
*   **ITEM 87**: The subject tells stories in which others are saying positive things about him or herself.
*   **ITEM 133**: The subject takes pleasure in referring a lot to his or her own positive but superficial attributes, like being beautiful, lovable, smart, well-dressed, worthy, a center of attention. This may be true even if the subject longs for qualities that are only imagined, wished for, or in the past.
*   **ITEM 135**: When confronted with problems, the subject prefers to dwell on his or her own positive qualities, such as being lovable, smart, beautiful, creative, "the best," as if those qualities will take care of the problems.
*   **Idealization of other's image**:
*   **ITEM 16**: The subject makes many references to how important certain people or objects are with an emphasis on their image, rather than real abilities or accomplishments which might make the person or object important to others.
*   **ITEM 17**: The subject tells stories in which he or she says glowing things about others.

positive things about another person or object, without giving much detail to back it up.

■ **ITEM 95:** When confronted with problems, the subject prefers to dwell on the positive qualities of others on whom he or she relies, such as being lovable, smart, beautiful, creative, ''the best,'' as if those qualities will take care of the problems.

■ **ITEM 138:** The subject takes pleasure in referring a lot to positive but superficial attributes of others, like being beautiful, lovable, smart, well-dressed, worthy, a center of attention. This may be true even if the subject longs for qualities that are only imagined, wished for, or in the past.

■ **ITEM 139:** When confronted with any negative aspects of others important to the subject, the subject appears to downplay or ignore them, by substituting talk about the positive image or attributes instead.

*   ○ **Distinction:** Less severe than Splitting. Idealization is an "exaggerated positive evaluation" but generally retains some reality testing.

*   **Omnipotence**

*   ○ **Definition:** Responding to emotional conflict or stressors by acting superior to others, as if one possessed special powers or abilities.

*   ○ **Example:** A person facing a complex problem they cannot control asserts with excessive bravado, "Don't worry, I can handle anything."

*   ○ **DMRS-Q Items:**

■ **ITEM 7:** The subject talks about how capable he or she is of influencing events or famous and important people. However, the emphasis is on the sense of personal power or abilities, rather than the detailed stories that support the claims as real.

■ **ITEM 10:** The subject acts in a very self-assured way and asserts an 'I can handle anything' attitude, in the face of problems that he or she in fact cannot fully control.

■ **ITEM 68:** The subject makes clearly false statements about his own special powers and abilities (these may or may not be delusional).

■ **ITEM 126:** There is excessive bravado in discussing problems or personal accomplishments that stands out as excessive or unrealistic.

■ **ITEM 129:** The subject is very grandiose in describing personal plans, accomplishments or abilities, perhaps comparing him or herself to famous people.

### Level 5: Neurotic Defenses

This level involves keeping unacceptable wishes or thoughts out of awareness, often by separating the cognitive and emotional components.

* **Repression**
* **Definition:** Being unable to remember or be cognitively aware of disturbing wishes, feelings, thoughts, or experiences.
* **Example:** A person experiences a sudden wave of sadness but has no conscious idea what caused it.
* **DMRS-Q Items:**
* **ITEM 13:** The subject keeps unpleasant things vague: he or she has trouble remembering or can't recall specific examples, when at least some should be forthcoming. This may include loss of memory for whole periods of time (e.g., childhood).
* **ITEM 47:** At points when a topic is emotionally loaded, the subject forgets what he or she is talking about and seems to get lost while talking.
* **ITEM 50:** When discussing a topic that brings up negative, conflicting feelings, the subject prefers to keep things vague, reflected in very vague, general or inexact statements.
* **ITEM 108:** The subject cannot remember certain facts which would normally not be forgotten, such as a distressing incident, reflecting some uneasy feelings about the topic.
* **ITEM 136:** When certain feelings or wishes would arise, the subject gives some evidence of them such as crying or appearing anxious but cannot clearly identify in words the specific feeling or the specific ideas that give the wish a clear meaning.
* **Distinction:** Characterized by "affect without idea," the converse of Isolation of Affect ("idea without affect").
* **Dissociation**
* **Definition:** A temporary alteration in the integrative functions of consciousness or identity to deal with emotional conflict.
* **Example:** A person describes feeling as if they were watching themselves in a movie during a stressful event (depersonalization).
* **DMRS-Q Items:**
* **ITEM 8:** The subject behaves or says something in a very uncharacteristic way that expresses an uninhibited impulse operating out of the subject's usual control, yet the subject is surprised by it (e.g., "I threw a glass of water in my friend's face, but I don't know what made me do it").
* **ITEM 27:** The individual describes fugue states, amnesia (not alcoholic blackouts), multiple personality, spontaneous trance states, or temporary loss of sensory or motor function.
* **ITEM 30:** In response to an emotionally charged situation, the subject suddenly becomes confused, depersonalized, "spaced out," or can't think or talk about the topic. Consciousness becomes clouded to a lesser or greater extent.

*   **ITEM 41:** In response to a distressing topic or situation, the subject develops a symptom, such as headache, stomach pain, or loss of an ability to do something, which temporarily eclipses awareness of what was distressing. The symptom may have a symbolic relationship to the type of distress.

*   **ITEM 73:** The subject associates with or is fascinated by people who do very uninhibited, dramatic, or socially outrageous things, which appear to express some of the subject's own inhibited wishes. Nonetheless, the subject is unaware of any such connection.

*   **Reaction Formation**

*   **Definition:** Substituting behavior, thoughts, or feelings that are diametrically opposed to one's own unacceptable thoughts or feelings.

*   **Example:** An employee who harbors deep-seated hostility toward their boss behaves in an overly respectful and sycophantic manner.

*   **DMRS-Q Items:**

*   **ITEM 52:** When confronting a personal wish about which the subject may feel guilty, the subject does not acknowledge or express it, but substitutes an opposite attitude against the wish, for instance, a desire is supplanted by renunciation or anger at anything to do with the desire.

*   **ITEM 55:** The subject is very compliant, agreeing to most everything the interviewer points out, when some disagreement and discussion would be expected.

*   **ITEM 74:** In dealing with people who are angry or abusive, the subject is cooperative and nice and eager to please, failing to express any negative feelings which might be expected.

*   **ITEM 96:** In relationships, the subject has an attitude of giving much more than he or she receives but is unaware of the imbalance.

*   **ITEM 99:** In fearful situations, the subject does not show expected fear, but reacts with exaggerated enthusiasm or courage, failing to acknowledge the fear.

*   **Displacement**

*   **Definition:** Redirecting a feeling or response from its original object to another, usually less threatening, object.

*   **Example:** After being reprimanded by their boss, a person comes home and yells at their child for a minor issue.

*   **DMRS-Q Items:**

*   **ITEM 1:** In dealing with an important problem that makes the anxious, the subject prefers to focus on minor or unrelated matters instead, which distracts the subject away from the central problem, for example, cleaning or organizing rather than working on projects that need to be done.

*   **ITEM 64:** The subject directs strong feelings toward a person or object who has little to do with the subject but who may bear similarities to

someone significant to the subject. The subject may be somewhat puzzled by the reason for the strength of these feelings.

■ **ITEM 69:** When confronting emotionally charged topics, the subject tends not to address concerns directly and fully but wanders off to tangentially related topics that are emotionally easier for the subject to discuss or prefers to pay attention to someone else dealing with a similar situation. This can include preferring to read or watch a film portraying people dealing with similar problems.

■ **ITEM 122:** When discussing an affect-laden event, the subject expresses more feelings directed toward incidental details or issues than about the major point or effect of the event, perhaps appearing "picky."

■ **ITEM 125:** The subject gets irritated easily by minor things that bother him or her and tends to lose a focus on the main things that need attention.

○ **Distinction:** Differentiated from Projection. In displacement, the person is aware of their emotion but directs it at the wrong target. In projection, the person is unaware of their emotion and attributes it to someone else.

# Level 6: Obsessional Defenses

This level involves managing anxiety by exerting excessive intellectual control over threatening feelings.

* **Isolation of Affect**

* ○ **Definition:** Being unable to simultaneously experience the cognitive and affective components of an experience because the affect is kept from consciousness.

* ○ **Example:** A car accident survivor describes the event in precise, objective detail but, when asked how they felt, says "I didn't feel anything."

* ○ **DMRS-Q Items:**

* ■ **ITEM 28:** When telling an emotionally meaningful story, the subject states that he or she does not have specific feelings that one would expect, although the subject recognizes that he or she should feel something.

* ■ **ITEM 31:** In talking about a meaningful, emotionally charged experience, the subject talks in a detached way, as if he or she is not in touch with the feelings that should surround it.

* ■ **ITEM 39:** The subject clearly describes the details of either positive or distressing or traumatic experiences but fails to show any attendant emotion in tone of voice, facial expression, or bodily expression.

* ■ **ITEM 107:** The subject talks as if emotionally detached from whatever he says about himself or his experiences.

*   **ITEM 140**: The subject describes events with good detail, but without mention of any attendant feelings, like a reporter describing the narrative of someone's life, but devoid of personal reactions.

*   **Distinction**: The converse of Repression. Isolation is "idea without affect," while Repression is "affect without idea."

*   **Intellectualization**
    *   **Definition**: The excessive use of abstract thinking, generalization, and logic to avoid disturbing feelings.
    *   **Example**: When asked about a tense relationship, a person responds, "From a psychosocial perspective, this is a product of intergenerational trauma," thus avoiding the actual feelings of pain.
    *   **DMRS-Q Items**:
        *   **ITEM 4**: When confronting personal issues, the subject tends to ask general questions, as if getting general information or answers from others will elucidate his or her own feelings and concerns. As a result, personal reactions are kept at a distance.
        *   **ITEM 26**: The subject talks about his personal experiences by making general statements that appear accurate but somehow avoid revealing specific personal feelings and reactions.
        *   **ITEM 53**: There is a lifeless quality to most of the subject's descriptions of his feelings and reactions, because the subject tries to explain them intellectually rather than experience or express them. For example: 'My present predicament is an inevitable product of my parents' extreme expectations and other parental experiences when growing up.'
        *   **ITEM 57**: The subject distances him or herself from his or her own feelings by speaking about him or herself in the second or third person a lot, as if the subject were talking about someone else.
        *   **ITEM 60**: Whenever focusing on personal issues or experiences the subject tends to generalize or even discuss things in a logical or scientific way, thereby keeping his feelings and experiences very distant.

*   **Undoing**
    *   **Definition**: Behavior designed to symbolically make amends for or negate previous thoughts, feelings, or actions that provoke guilt.
    *   **Example**: After saying something harsh to their spouse, a person feels uneasy but does not apologize. The next day, they buy an expensive gift, as if to "undo" the previous hurtful words.
    *   **DMRS-Q Items**:
        *   **ITEM 48**: When another person tries to clarify a statement made by the subject, the subject says thing like 'well, not really' or 'not exactly' followed by qualifications that do not clearly clarify things. Because the subject is wary of committing him or herself to any statement, the listener may be unsure as to the subject's definite opinion.
        *   **ITEM 67**: The subject spontaneously describes some of his or her

actions which are followed by actions that are of the opposite intent, as if every action must be balanced by an equal but opposite action. The subject is aware of the contradiction which may seem vexing or ironic.

*   **ITEM 70:** The subject prefaces a strong statement about a topic with a disclaimer, to the effect that what he or she is about to say may not be true.

*   **ITEM 81:** The subject conveys opinions about something or someone with a series of opposite or contradictory statements, as if uncomfortable with taking a clear stand one way or the other.

*   **ITEM 83:** After the subject has done something that probably results in a feeling of guilt or shame, the subject makes an act of reparation, as if sorry. However, the subject focuses on the act but avoids dealing with the sense of guilt or shame as one would whenever making a normal apology.

## Level 7: High-Adaptive (Mature) Defenses

This level involves handling stressors in the most adaptive ways, integrating feelings with ideas and seeking resolutions.

*   **Affiliation**
    *   **Definition:** Turning to others for help and support to collaboratively find solutions.
    *   **Example:** After making a mistake at work, an individual calls a trusted colleague to talk through the situation and get advice.
    *   **DMRS-Q Items:**
        *   **ITEM 22:** Whenever the subject brings a personal problem to someone for help or advice, the subject is not expecting the other to take care of it, but rather to help come up with a solution which the subject will then implement.
        *   **ITEM 25:** The subject describes an important conflict or external stress in which affiliation played a major emotional role in coping as evident by the description of characteristics of the help received, the individuals or organization involved, and the sense that something was taken away from the experience.
        *   **ITEM 44:** When the subject describes seeking help from others, there is a sense of having learned something from the interchange.
        *   **ITEM 66:** When confronted with emotional conflict or stressful situations, the subject describes confiding in someone. Emotionally meaningful sharing led to enhancement of coping skills, or direct assistance beyond what the subject would have done alone.
        *   **ITEM 93:** When dealing with an emotionally difficult situation, the subject reports that talking to others helps the subject think through

how best to handle the problem.

* **Altruism**

* **Definition**: Dedication to fulfilling the needs of others as a way of dealing with one's own emotional conflicts.

* **Example**: A person who has recovered from a major loss volunteers to support others going through the same experience.

* **DMRS-Q Items**:

* **ITEM 11**: The subject helps others who are experiencing a problem they cannot adequately deal with alone. The problem appears to have a personal meaning to the subject related to similar experiences in the subject's past (e.g., 'It made me feel good to help someone in the same position that I once found so difficult'.).

* **ITEM 15**: The subject finds it personally rewarding to help others who are suffering.

* **ITEM 79**: The subject participates in organizations or groups that help other people in direct person-to-person ways. In this context, the subject gives direct help to others, which the subject apparently finds rewarding.

* **ITEM 104**: The subject reacts to a difficult or dangerous situation for someone else by interposing him or herself to protect the other person. While not reckless, the subject may put him or herself at personal physical or material risk in doing so.

* **ITEM 132**: The subject helps others who are at a loss to cope with a problem or situation, possibly including standing up to authority, It is clear that the subject obtains some personal gratification or mastery from the meaning of helping, beyond any overt reward obtained.

* **Anticipation**

* **Definition**: Mitigating stressors by considering realistic outcomes and mentally rehearsing potential emotional reactions to future problems.

* **Example**: Before a major presentation, a person spends time imagining difficult questions and planning their responses to reduce anxiety.

* **DMRS-Q Items**:

* **ITEM 43**: Ahead of an important performance or occasion, the subject practices imagining him or herself in the situation to be both better prepared and less anxious.

* **ITEM 46**: The subject describes small events in his or her life in which he or she characteristically mentions thinking about their outcomes ahead of time and emotionally preparing in some way for them.

* **ITEM 62**: In confronting a new situation or an unknown task, the subject tries ahead of time to be aware of the emotional challenges and plan for whatever resources that will aid and comfort the subject in the new situation.

* **ITEM 65**: The subject describes emotionally meaningful vignettes of

upcoming stressful situations in which the subject fully prepared him or herself emotionally as to how to handle it.
*   ■ **ITEM 78:** In dealing with interpersonal conflicts, the subject tries to imagine how others might respond in planning how to deal with them, but without obsessing or over planning.
*   **Humor**
- **Definition:** Emphasizing the amusing or ironic aspects of a conflict to relieve tension, without being at anyone's expense.
- **Example:** In a tense meeting, a person makes a self-deprecating but relevant joke that allows everyone to acknowledge a shared difficulty.
- **DMRS-Q Items:**
- ■ **ITEM 18:** The subject makes amusing or ironic comments about embarrassing situations to diffuse them.
- ■ **ITEM 37:** The subject can make humorous remarks about him or herself or others without saying negative, hurtful, or deprecating things.
- ■ **ITEM 40:** In confronting difficult situations which the subject cannot change, the subject uses humor about the situation to mitigate the negative feelings arising.
- ■ **ITEM 51:** The subject diffuses a difficult situation with others by making a pertinent joke that centers on some important point that all can acknowledge without being at anyone's expense, thereby fostering cooperation.
- ■ **ITEM 119:** When confronted by a situation fraught with competitive, hostile, or jealous feelings, the subject reveals something about him or herself in a self-deprecatory, ironic, or amusing way to diffuse the tension.
*   **Self-Assertion**
- **Definition:** Expressing one's feelings, thoughts, and needs directly and respectfully to achieve goals.
- **Example:** When interrupted, a person says politely but firmly, "Please let me finish my point, then we can discuss your thoughts."
- **DMRS-Q Items:**
- ■ **ITEM 23:** When pursuing something desirable, including a relationship with someone, the subject can use his or her talents and charms to attract the other, without feeling ashamed or guilty if unsuccessful.
- ■ **ITEM 90:** When the subject has a physical or emotional or practical problem, the subject takes steps to deal with his or her needs possibly including initiating getting help - rather than ignore them or hope they will take care of themselves.
- ■ **ITEM 105:** When someone is impolite, dismissive, or derogatory toward the subject, the subject can stand up for him or herself appropriately, even if the subject cannot change the other's attitude or command an apology.

*   **ITEM 109:** The subject can disagree with others and express opinions without being overly hostile, devaluing, or manipulative of others.
*   **ITEM 146:** When confronted with emotionally difficult situations, the subject expresses his or her thoughts, wishes, or feelings clearly and directly without inhibition or excess.
*   **Self-Observation**
*   **Definition:** Reflecting on one's own thoughts, feelings, motivation, and behavior to gain insight.
*   **Example:** After an argument, a person reflects, "Why did I get so angry? Perhaps their words triggered my fear of being ignored."
*   **DMRS-Q Items:**
*   **ITEM 9:** When talking with someone about a personally charged topic, the subject displays an accurate view of him or herself and can see how he or she appears from the other person's point of view.
*   **ITEM 32:** When confronting emotionally important problems, the subject can reflect upon relevant personal experiences and explore emotional reactions. This allows the subject to adjust better to limitations and compromises, possibly leading to more fulfilling outcomes.
*   **ITEM 58:** In interpersonal conflicts, the subject uses an understanding of his or her reactions to facilitate understanding others' points of view or subjective experiences. This may make the subject a better negotiator or collaborator.
*   **ITEM 77:** When considering an emotionally important decision, the subject explores his or her own motives and limitations to arrive at a more fulfilling decision.
*   **ITEM 91:** When the subject reflects on past experiences, he or she can relive distressing feelings and make connections between events and feelings and develop understanding thereby changing how the subject views the past and possibly similar situations in the present.
*   **Sublimation**
*   **Definition:** Channeling potentially maladaptive feelings or impulses into socially acceptable behaviors, such as artistic or athletic pursuits.
*   **Example:** After a painful breakup, an individual channels their feelings of sadness into writing poetry or training for a marathon.
*   **DMRS-Q Items:**
*   **ITEM 14:** In describing any personal artistic or creative activities - such as writing, music, art, or acting-the subject appears to transform emotional conflicts or unfulfilled wishes from elsewhere in life, helping to shape the creative activity or product.
*   **ITEM 36:** The subject describes emotional conflictual situations in which some of the feelings or dissatisfaction are channeled into creative or artistic activities. The resulting creative products such as a poem or painting - give the subject a sense of mastery or relief from the

conflicts.
*   **ITEM 63:** Whenever engaging in a creative activity, the subject finds the process of creation itself satisfying, apart from any satisfaction with the final product.
*   **ITEM 97:** Following experiences of emotional distress or conflict, the subject engages in sports or other physical activities which are an invigorating outlet for any lingering frustrations.
*   **ITEM 100:** Following some strong experiences, the subject engages in his or her ordinary activities but with less effort, greater accomplishment and more pleasure than they normally would require or yield.
*   **Suppression**
*   **Definition:** Voluntarily and temporarily avoiding thinking about disturbing problems to attend to a more pressing task. This is postponing, not denying.
*   **Example:** Upon receiving an urgent work task, a person consciously decides, "I'll focus on this now and deal with my other worries this evening."
*   **DMRS-Q Items:**
*   **ITEM 49:** When presented with an external demanding situation over which the subject has no control, the subject can accept the demand, putting negative feelings aside to deal with what must be done.
*   **ITEM 117:** When the subject experiences a desire that if acted upon would have bad consequences, the subject is able to decide consciously to put the desire aside and not act upon it.
*   **ITEM 128:** When the subject experiences a salient personal limitation or problem, rather than pretending it is not a problem, the subject acknowledges and accepts it, which allows the subject to avoid exacerbating problems. For example, acknowledging an addiction and accepting that one must avoid using the desired substance.
*   **ITEM 131:** When attending to something emotionally important, if interrupted by something more urgent, the subject attends to the interruption as needed, but later returns and finishes dealing with what had to be postponed.
*   **ITEM 150:** When presented with an emotionally charged situation, the subject can postpone dealing with his or her feelings to attend to the things that need to be done immediately. The feelings don't get in the way or distract the subject, because the subject is able to give them adequate attention later.

# Part 3: FAQ and Special Cases

1. **What if a single sentence contains multiple defense mechanisms?**
*   **Principle:** Choose the most central, dominant defense mechanism. If it's truly impossible to

distinguish and multiple defenses are prominent, consider using Level 8: Unclear / Needs More Information and make a note. If multiple defenses are present, the one from a lower (less mature) level is typically chosen as the primary label.

**2. Which defenses are difficult to identify in short conversations?**

*   **Splitting:** The core evidence for splitting is contradictory, absolute views of the same person at different times ("all-good" then "all-bad"). A short dialogue will likely only capture one side (e.g., only idealization or only devaluation), making it impossible to observe the shift.

*   **Reaction Formation:** We need sufficient reason to believe that the expressed behavior (e.g., excessive kindness) is the opposite of a deeply held, unacceptable unconscious impulse (e.g., hostility). Without knowing a person's long-term patterns, this is difficult to confirm.

*   **Repression:** True repression is unconscious, so its content will not appear in the dialogue. We can only infer it from its traces, like sudden forgetfulness or vague memories. This is a "deduction about an absence," which is very difficult to substantiate in short texts.

*   **Undoing:** Undoing is a sequence of "Act A + Act B," where a symbolic act of reparation neutralizes a preceding guilt-inducing act. A short conversation may only capture one part of the sequence.

*   **Dissociation & Autistic Fantasy:** We typically identify these by hearing a report of a past experience ("I felt like I was outside my body," or "I spent the afternoon daydreaming...") rather than observing them in real-time.

**3. Is all self-expression considered Self-Assertion?**

*   No. Self-Assertion is about directly expressing feelings and thoughts to achieve a goal in the face of an emotional conflict or stressor. Its function is to relieve the anxiety of suppressing one's wishes or to avoid the shame of not speaking up for oneself. It's an active coping behavior in a challenging situation.

*   **Example Distinction:** A client telling a therapist, "I was very sad when I broke up with my girlfriend," is not self-assertion. This is **Self-Disclosure**—reporting a past emotional experience in a safe environment. Self-assertion would be saying to the partner *at the moment of the breakup*, "Your unilateral decision makes me feel very hurt and disrespected, and I need to express how I feel right now." That is using direct expression to navigate an immediate conflict.

**4. Is all negative talk about others a defense?**

*   Not necessarily. Consider the following:

*   ○ **Intent:** Is the purpose to elevate oneself by demeaning others, shift blame, or avoid guilt (a defense)? Or is it to state a fact, solve a problem, or communicate directly (not a defense, or a mature defense)?

o **Reality Distortion:** Is the negative evaluation exaggerated, a misattribution of blame, or unjust (a defense)? Or is it factually based (not a defense)?
o **Context:** Is the statement made in response to a threat to self-esteem, an internal conflict, or stress (likely a defense)?

5. **How should a simple "thank you" be considered?**

*   A simple "thank you" on its own is not a defense (Level 0). However, a sincere and detailed expression of gratitude, especially one that reflects on the help received and builds connection, can be an instance of Affiliation (Level 7).

**Situations That Are Typically NOT Defenses (Level 0)**

*   Greetings and social pleasantries.
*   Simple expressions of thanks.
*   Simple responses like "yes," "no," or "okay."
*   Reporting factual details of an event.
*   Asking factual questions (e.g., "When is our next appointment?").
*   Social small talk on neutral topics like weather or traffic.
*   Expressing direct physiological sensations (e.g., "I feel a bit cold").
*   Expressing genuine confusion for clarification (e.g., "I'm sorry, I don't understand what you mean").
*   Describing objective personal information (e.g., "I am 30 years old").
*   Acknowledging a therapist's interpretation as accurate.