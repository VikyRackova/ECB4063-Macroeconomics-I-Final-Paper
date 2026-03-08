//****************************************************************************/
/*																			*/
/* This file aggregates the individual tax data into a shocks series 		*/
/* "Discretionary Tax Changes and the Macroeconomy: New Narrative Evidence  */
/*       from the United Kingdom"											*/
/*	The American Economic Review, June 2013 								*/
/*	https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1507				*/
/*																			*/
/* Author: James Cloyne, American Economic Review, June 2013				*/
/*																			*/
/*																			*/
/*																			*/
//****************************************************************************/

// First load GDP data
clear
set more off

cd ""

global datadir ""

import excel "${datadir}/CloyneNarrativeDataset.xlsx", sheet("Nominal GDP") cellrange(B3:G264) firstrow

gen Date = yq(Year, Quarter) 

keep Date Year Quarter NominalGDP

format Date %tq

tempfile GDPdata

save "`GDPdata'"

keep Date

tempfile RawDates

save "`RawDates'"

// Now create tax shock data

clear

// Important data and format columns
cd ""

import excel "${datadir}/CloyneNarrativeDataset.xlsx", sheet("TaxData") cellrange(A1:P2463) firstrow

ren 	Date BudgetDate
drop 	if ImplementationDate == .

drop 	if Excluded == 1

format	ImplementationDate %td

gen 	Announcement  =	date(AnnouncementDate,"MDY")

format	Announcement %td

replace TaxData = "0" if TaxData == "" | TaxData == "*"
destring TaxData, generate(Taxseries)


// Exclude retroactive changes
gen 	AdjustedImplementation = ImplementationDate + 45
format 	AdjustedImplementation %td

gen 	AdjustedAnnouncement = Announcement + 45
format 	AdjustedAnnouncement %td

gen 	ImplementationDate_ExclRetro = ImplementationDate
replace	ImplementationDate_ExclRetro = Announcement if ImplementationDate < Announcement
format 	ImplementationDate_ExclRetro %td

gen 	AdjustedImplDate_ExclRetro = ImplementationDate_ExclRetro + 45 // Actions taking place in the second half of the quarter are assigned to the next quarter
format 	AdjustedImplDate_ExclRetro %td

encode Major, g(MotiveMajor) l(MajorMotive)
encode Minor, g(MotiveMinor) l(MinorMotive)


// 1. Need to deal with retroactive measures

tempfile PreCollapse

save "`PreCollapse'"

local var ExclRetro

use "`PreCollapse'", clear

gen 	Quarter 	= quarter(AdjustedImplDate_`var')
gen 	Year 		= year(AdjustedImplDate_`var')

collapse (sum) Taxseries, by(MotiveMajor Year Quarter)

sort Year Quarter MotiveMajor 

// Need to merge in GDP data

gen Date = yq(Year, Quarter) 
format Date %tq

xtset MotiveMajor Date

tsfill

replace Taxseries = 0 if Taxseries == .

merge m:1 Date using "`GDPdata'", nogen keep(3)

// First construct major categories
foreach motive in N X {

	gen `motive'_TaxToGDP = Taxseries*100/(NominalGDP)

	sort MotiveMajor Date
	
	preserve 
	
	keep if MotiveMajor == "`motive'":MajorMotive

	keep Date `motive'_TaxToGDP

	local 	motiveFile = "`motive'File"
	di "`motiveFile'"
	tempfile `motiveFile'
	save ``motiveFile''
	
	restore
	
}
use "`PreCollapse'", clear
	
gen 	Quarter 	= quarter(AdjustedImplDate_`var')
gen 	Year 		= year(AdjustedImplDate_`var')

collapse (sum) Taxseries, by(MotiveMinor Year Quarter)

sort Year Quarter MotiveMinor 

// Need to merge in GDP data

gen Date = yq(Year, Quarter) 
format Date %tq

xtset MotiveMinor Date

tsfill

replace Taxseries = 0 if Taxseries == .

merge m:1 Date using "`GDPdata'", nogen keep(3)

// First construct major categories
foreach motive in DM SS DR SD LR IL DC ET {

	gen `motive'_TaxToGDP = Taxseries*100/(NominalGDP)

	sort MotiveMinor Date
	
	preserve 
	
	keep if MotiveMinor == "`motive'":MinorMotive

	keep Date `motive'_TaxToGDP

	local 	motiveFile = "`motive'File"
	di "`motiveFile'"
	tempfile `motiveFile'
	save ``motiveFile''
	
	restore
	
}

use "`RawDates'", clear

foreach motive in N X DM SS DR SD LR IL DC ET  {
	
	local motiveFile = "`motive'File"
	merge 1:1 Date using ``motiveFile'', nogen
	
	replace `motive'_TaxToGDP = 0 if `motive'_TaxToGDP == .
	
}

export excel using "${datadir}/CloyneAERNarrativeTaxShocks.xlsx", replace first(var)


